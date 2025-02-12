# **Calibrated Decision-Making through Large LLM-Assisted Retrieval**

This repository provides the implementation of [Calibrated Decision-Making through LLM-Assisted Retrieval](https://arxiv.org/abs/2411.08891).

### Abstract
<sub>Recently, large language models (LLMs) have been increasingly used to support various decision-making tasks, assisting humans in making informed decisions. However, when LLMs confidently provide incorrect information, it can lead humans to make suboptimal decisions. To prevent LLMs from generating incorrect information on topics they are unsure of and to improve the accuracy of generated content, prior works have proposed Retrieval Augmented Generation (RAG), where external documents are referenced to generate responses. However, traditional RAG methods focus only on retrieving documents most relevant to the input query, without specifically aiming to ensure that the human user's decisions are well-calibrated. To address this limitation, we propose a novel retrieval method called Calibrated Retrieval-Augmented Generation (CalibRAG), which ensures that decisions informed by the retrieved documents are well-calibrated. Then we empirically validate that CalibRAG improves calibration performance as well as accuracy, compared to other baselines across various datasets.</sub>

---

## **Prerequisites**

All experiments were conducted on a single NVIDIA RTX A6000 GPU.

```bash
cd code
conda create -n calibrag python=3.9 -y
conda activate calibrag
pip install -r requirements.txt
```

---

## **Synthetic Data Generation**

### **1. Create Open-ended Questions**
```bash
python -m experiments.api --data_dir ./data/dev/raw --type oe
```

### **2. Generate RAG Data**

1. **Download Preprocessed Passage Data**   
   ```bash
   wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
   ```

2. **Run Retrieval**  
    ```bash
    sh scripts/retrieve_dev.sh
    ```

### **3. Generate LLM Outputs**
For baselines,
```bash
sh scripts/base_calibrag_lm_outputs.sh
```
For CalibRAG,
```bash
sh scripts/calibrag_lm_outputs.sh
```

### **4. Simulate Human Decision-Making**
```bash
sh scripts/calibrag_decision.sh
```

### **5. Evaluate Results**
For baselines,
```bash
sh scripts/base_train_api.sh
```
For CalibRAG,
```bash
sh scripts/calibrag_api.sh
```

<p>
  Access the finalized data files here: 
  <a href="https://huggingface.co/datasets/yuntokki/calibrag">
    <img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="Hugging Face" style="width: 20px; vertical-align: middle;"> CalibRAG
  </a>
</p>

---

## **Training Methods**

### **1. CT-LoRA**
```bash
python -m experiments.train.train_calibration_tune \
  --model_name="Meta-Llama-3.1-8B-Instruct" \
  --batch_size 2 \
  --gradient_accumulation_steps 2 \
  --uc_type "ct"
```

### **2. CT-Probe**
```bash
python -m experiments.train.train_classifier_tune \
  --model_name "Meta-Llama-3.1-8B-Instruct" \
  --batch_size 4
```

### **3. CT-Ling (Sampling)**
```bash
python -m experiments.train.train_calibration_tune \
  --model_name="Meta-Llama-3.1-8B-Instruct" \
  --batch_size 2 \
  --gradient_accumulation_steps 2 \
  --uc_type "ling"
```

### **4. CT-Number (Sampling)**
```bash
python -m experiments.train.train_calibration_tune \
  --model_name="Meta-Llama-3.1-8B-Instruct" \
  --batch_size 2 \
  --gradient_accumulation_steps 2 \
  --uc_type "number"
```

### **5. CalibRAG Training**
```bash
python -m experiments.train.train_calibrag \
  --model_name="Meta-Llama-3.1-8B-Instruct" \
  --batch_size 2 \
  --gradient_accumulation_steps 2 \
  --with_lora True
```

---

## **Test Data Generation**

### **1. Create Open-ended Questions**
```bash
    python -m experiments.api --data_dir ./data/test/raw --type oe
```

### **2. Generate RAG Data**
```bash
   sh scripts/retrieve_test.sh
```

### **3. Produce LLM Outputs with Uncertainty**
For baselines,
```bash
sh scripts/base_eval_lm_outputs.sh
```
For CalibRAG,
```bash
sh scripts/calibrag_eval_lm_outputs.sh
```

### **4. Simulate Human Decision-Making for Testing**
For baselines,
```bash
sh scripts/base_eval_decision.sh
```
For CalibRAG,
```bash
sh scripts/calibrag_decision.sh
```

### **5. Evaluate Results**
For baselines,
```bash
sh scripts/base_eval_api.sh
```
For CalibRAG,
```bash
sh scripts/calibrag_api.sh
```

---

## **References**
```bash
https://github.com/facebookresearch/contriever
https://github.com/tatsu-lab/linguistic_calibration
https://github.com/activatedgeek/calibration-tuning
https://github.com/esteng/pragmatic_calibration
```