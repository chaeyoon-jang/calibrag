import os
import logging
import torch
import torch.nn as nn

from .utils import get_last_checkpoint_path

def get_classifier(input_size=None, output_size=None, bias=False, **_):
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

    return model

def get_classifier_head(
    input_size=None,
    classifier_model_name="mlp_binary",
    checkpoint_dir=None,
    is_trainable=False,
    weights_name="classifier_model.bin",
    output_size=2
):
    classifier_model = get_classifier(
        input_size=input_size, output_size=output_size
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
