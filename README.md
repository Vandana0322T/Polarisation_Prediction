# Detects polarized text (binary: 0/1) across 9 languages using a clean Hugging Face BERT baseline.
# Project Structure
polarization-baseline/
├─ subtask1/
│  ├─ train/                
│  └─ dev/                  
├─ src/
│  └─ train_bert.py         
├─ outputs/                 
├─ requirements.txt
└─ README.md

Eval = non-overlapping slice of train.
Dev = predictions only

# Data format (CSV columns)
id (string/int) – unique per row
text (string)
polarization (0 or 1) – required for train

# Dataset (9 langs)
subtask1/train/{amh,arb,deu,eng,hau,ita,spa,urd,zho}.csv
subtask1/dev/{amh,arb,deu,eng,hau,ita,spa,urd,zho}.csv

# Environment Set Up
python -m venv .venv
source .venv/bin/activate        
pip install -r requirements.txt

# To train and run
python -m src.train_bert \
  --data_root subtask1 --train_dir train --dev_dir dev \
  --epochs 3 --lr 2e-5 --train_bs 64 --eval_bs 8 \
  --val_from_train 0.1 --seed 42

# For each language:
Train on train/lang.csv using BERT (eng: bert-base-uncased; others: bert-base-multilingual-cased)
Eval on a held-out stratified slice of train (no overlap)
Predict on dev/lang.csv (ids preserved)
Results are saved under outputs/<lang>/.

# Outputs
**Per language (e.g., outputs/eng/):**
metrics.json – macro F1 / precision / recall (from held-out train)
predictions_dev.csv – columns: id,prediction (for dev)
split_ids.json – exact train/val IDs used (reproducible)
training_log.txt
**All languages:**
outputs/summary_metrics.csv – one row per language



