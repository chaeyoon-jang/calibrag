import os
import logging

import math
import torch
import torch.nn as nn

from .utils import get_last_checkpoint_path

class TempFourier(nn.Module):
    def __init__(self, K, t_range=(1.0, 2.0)):
        super().__init__()
        t_min, t_max = t_range
        w_base = 2 * math.pi / (t_max - t_min)
        freqs = (2 ** torch.arange(K, dtype=torch.float32)) * w_base       
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        ft = t.unsqueeze(1) * self.freqs.to(t.dtype)               
        return torch.cat([ft.sin(), ft.cos()], dim=1)  
    
class PEClassifier(nn.Module):
    def __init__(self,
                 input_size,       
                 output_size,      
                 t_range=(1.0, 2.0),
                 mode="add",     
                 fourier_K=6,
                 bias=False):
        super().__init__()
        assert mode in {"add", "concat"}
        self.mode = mode

        self.fourier = TempFourier(fourier_K, t_range)
        if mode == "add":
            self.proj = nn.Linear(2*fourier_K, input_size, bias=False)

        in_dim = input_size if mode == "add" else input_size + 2*fourier_K
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 128),    
            nn.ReLU(),
            nn.Linear(128,  64),    
            nn.ReLU(),
            nn.Linear(64, output_size, bias=bias),
        )

    def forward(self, h, t):
        pe = self.fourier(t).to(h.dtype)     
        # (B, 2K)
        h_cond = h + self.proj(pe)\
            if self.mode=="add" else torch.cat([h, pe], dim=-1)
        return self.mlp(h_cond)                   

def get_classifier(
    classifier_model_name="base",
    input_size=None, 
    output_size=None, 
    bias=False, 
    **_):
    
    if classifier_model_name == "base":
        # (B, -1, 4096)
        # (B, seq_len, vocab_size)
        model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size, bias=bias),
            # nn.Linear(input_size, output_size, bias=bias),
        )
        
    elif classifier_model_name == "fourier":
        # (B, -1, 4096)
        # (B, seq_len, vocab_size)
        model = PEClassifier(
            input_size=input_size,
            output_size=output_size,
            t_range=(1.0, 2.0),
            mode="add",
            bias=bias
        )
    return model

def get_classifier_head(
    input_size=None,
    classifier_model_name="base",
    checkpoint_dir=None,
    is_trainable=False,
    weights_name="classifier_model.bin",
    output_size=2
):
    classifier_model = get_classifier(
        input_size=input_size, 
        output_size=output_size,
        classifier_model_name=classifier_model_name,
    )

    if checkpoint_dir is not None:
        checkpoint_dir = get_last_checkpoint_path(checkpoint_dir)

        if os.path.isfile(f"{checkpoint_dir}/{weights_name}"):
            classifier_model.load_state_dict(
                torch.load(f"{checkpoint_dir}/{weights_name}")
            )

            logging.info(f"Loaded classifier model checkpoint from '{checkpoint_dir}'.")
    else:
        for module in classifier_model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    if is_trainable:
        classifier_model = classifier_model.train().requires_grad_(True)
    else:
        classifier_model = classifier_model.eval().requires_grad_(False)

    return classifier_model
