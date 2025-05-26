# ✨Overview

![Framework](figures/Framework.png)

Code for our paper "CroSec: Cross-model Security Hardening for Multiple Code LLMs in One-time Training"

## Directory Structure

The directory structure of this repository is shown as below:

```
.
|-- data_train_val      # Dataset for training and validation 
|-- data_eval           # Dataset for evaluation
|-- results	            # Experimental results
|-- scripts             # Scripts for training and inference
|-- trained	            # Trained LoRA for security.
|-- setup_codeql.sh
```

# 🔨 Setup

```
conda create -n CroSec python==3.10
conda activate CroSec
pip install -r requirements.txt
./setup_codeql.sh
```

# 🚀 1. Get embeddings 

```
python ./scripts/get_emb.py
```

# 🚀 2. Get similarity metrix

```
python ./scripts/get_simi.py
```

# 🚀 Train

To train a LoRA for security model, run:

```
python train_lora_sec.py
```

**We provide a trained LoRA plugin for Codegen to replicate our experiments.** You can download it from Google Drive: 

# 🚀 TEST

To test the Secutiy, run:



To test the Functional Correctness, run: