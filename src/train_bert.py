

# src/train_bert.py
import os, json, argparse, warnings, inspect, gc
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# Make wandb optional (keep starter behavior without dependency)
try:
    import wandb
    wandb.init(mode="disabled")
except Exception:
    wandb = None

# Try to import the official collator; otherwise use a tiny fallback
try:
    from transformers import DataCollatorWithPadding
    def make_collator(tokenizer):
        return DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
except Exception:
    def make_collator(tokenizer):
        # Minimal right-padding collator
        def simple_pad(batch):
            keys = list(batch[0].keys())
            max_len = 0
            for item in batch:
                if "input_ids" in item:
                    max_len = max(max_len, item["input_ids"].shape[0])
            out = {}
            for k in keys:
                tensors = []
                for item in batch:
                    t = item[k]
                    if k != "labels":
                        pad_len = max_len - t.shape[0]
                        if pad_len > 0:
                            pad_val = 0
                            t = torch.cat([t, torch.full((pad_len,), pad_val, dtype=t.dtype)])
                    tensors.append(t)
                out[k] = torch.stack(tensors)
            return out
        return simple_pad

LANGS = ["amh", "arb", "deu", "eng", "hau", "ita", "spa", "urd", "zho"]

# ----------------------------
# Data utilities
# ----------------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def has_labels(df: pd.DataFrame) -> bool:
    return "polarization" in df.columns

def clean_df(df: pd.DataFrame, expect_labels: bool) -> pd.DataFrame:
    # id
    if "id" not in df.columns:
        df["id"] = np.arange(len(df))
    # text
    df["text"] = df.get("text", "").fillna("").astype(str)

    if expect_labels:
        if "polarization" not in df.columns:
            raise ValueError("Expected labels but 'polarization' column is missing.")
        df["polarization"] = pd.to_numeric(df["polarization"], errors="coerce")
        df = df.dropna(subset=["polarization"])
        df["polarization"] = df["polarization"].astype(int)
        df = df[df["polarization"].isin([0, 1])]
    return df.reset_index(drop=True)

def split_train_val(df: pd.DataFrame, val_frac: float, seed: int, label_col="polarization"):
    if val_frac <= 0.0:
        raise ValueError("val_frac must be > 0.0 when using train-slice validation.")
    try:
        tr, va = train_test_split(
            df, test_size=val_frac, stratify=df[label_col], random_state=seed, shuffle=True
        )
    except Exception:
        tr, va = train_test_split(df, test_size=val_frac, random_state=seed, shuffle=True)
    return tr.reset_index(drop=True), va.reset_index(drop=True)

# Dataset

class PolarizationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels  # list[int] or None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]) if self.texts[idx] is not None else ""
        enc = self.tokenizer(
            text, truncation=True, padding=False, max_length=self.max_length, return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


# Metrics

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
    }


# Version-agnostic TrainingArguments

def make_training_args(out_dir, epochs, lr, train_bs, eval_bs, seed, disable_tqdm=False):
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters

    # MPS/CPU friendly caps
    train_bs = max(4, min(train_bs, 8))
    eval_bs  = max(4, min(eval_bs, 8))

    kwargs = dict(
        output_dir=out_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        logging_steps=100,
        seed=seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    if "gradient_accumulation_steps" in params:
        eff_target = 64
        kwargs["gradient_accumulation_steps"] = max(1, eff_target // train_bs)

    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"
    elif "do_eval" in params:
        kwargs["do_eval"] = True

    if "save_strategy" in params:
        kwargs["save_strategy"] = "no"
    elif "save_steps" in params:
        kwargs["save_steps"] = 0

    if "report_to" in params:
        kwargs["report_to"] = []

    if "disable_tqdm" in params:
        kwargs["disable_tqdm"] = disable_tqdm

    return TrainingArguments(**kwargs)


# Model choice

def pick_model_name(lang: str) -> str:
    # Keep starter spirit: English -> BERT uncased, others -> mBERT cased
    return "bert-base-uncased" if lang == "eng" else "bert-base-multilingual-cased"


# One-language runner

def run_one_language(
    lang: str,
    data_root: str,
    train_dir: str,
    dev_dir: str,
    epochs: int,
    lr: float,
    train_bs: int,
    eval_bs: int,
    val_from_train: float,
    seed: int,
    max_length: int = 128,
) -> Dict[str, float]:
    train_path = os.path.join(data_root, train_dir, f"{lang}.csv")
    dev_path   = os.path.join(data_root, dev_dir,   f"{lang}.csv")
    out_dir    = os.path.join("outputs", lang)
    os.makedirs(out_dir, exist_ok=True)

    # Load & clean
    train_df = clean_df(safe_read_csv(train_path), expect_labels=True)
    dev_df   = clean_df(safe_read_csv(dev_path),  expect_labels=False)  # predictions-only

    # Always evaluate on a non-overlapping slice of train
    effective_val = float(val_from_train) if val_from_train and val_from_train > 0 else 0.10
    tr_df, val_df = split_train_val(train_df, effective_val, seed)

    # Leakage guard
    train_ids = set(tr_df["id"].tolist())
    val_ids   = set(val_df["id"].tolist())
    assert train_ids.isdisjoint(val_ids), "Train/Val overlap detected!"
    with open(os.path.join(out_dir, "split_ids.json"), "w", encoding="utf-8") as f:
        json.dump({"val_from_train": effective_val,
                   "train_ids": sorted(list(train_ids)),
                   "val_ids": sorted(list(val_ids))}, f)

    # Tokenizer & model
    model_name = pick_model_name(lang)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.config.use_cache = False  # save memory

    # Datasets
    train_dataset = PolarizationDataset(
        tr_df["text"].tolist(), tr_df["polarization"].tolist(), tokenizer, max_length
    )
    eval_dataset = PolarizationDataset(
        val_df["text"].tolist(), val_df["polarization"].tolist(), tokenizer, max_length
    )

    # Training args + collator
    training_args = make_training_args(
        out_dir=out_dir, epochs=epochs, lr=lr, train_bs=train_bs, eval_bs=eval_bs, seed=seed
    )
    data_collator = make_collator(tokenizer)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # Log basics
    with open(os.path.join(out_dir, "training_log.txt"), "w", encoding="utf-8") as f:
        f.write(f"Lang: {lang}\nModel: {model_name}\n")
        f.write(
            f"epochs={epochs}, lr={lr}, "
            f"train_bs={training_args.per_device_train_batch_size}, "
            f"eval_bs={training_args.per_device_eval_batch_size}, "
            f"val_from_train={effective_val}, seed={seed}\n"
        )

    # Train & eval
    trainer.train()
    eval_results = trainer.evaluate()
    metrics = {
        "lang": lang,
        "f1_macro": float(eval_results.get("eval_f1_macro", np.nan)),
        "precision_macro": float(eval_results.get("eval_precision_macro", np.nan)),
        "recall_macro": float(eval_results.get("eval_recall_macro", np.nan)),
        "used_dev_labels": False,  # by design
        "val_from_train": float(effective_val),
        "model_name": model_name,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # ---- Predictions for dev (robust to any shape / empty dev) ----
    dev_texts = dev_df["text"].tolist()
    dev_ids   = dev_df["id"].tolist()

    class DevDataset(Dataset):
        def __init__(self, texts, tok, max_length=128):
            self.texts = texts; self.tok = tok; self.max_length = max_length
        def __len__(self): return len(self.texts)
        def __getitem__(self, idx):
            enc = self.tok(
                str(self.texts[idx]) if self.texts[idx] is not None else "",
                truncation=True, padding=False, max_length=self.max_length, return_tensors="pt"
            )
            return {k: v.squeeze(0) for k, v in enc.items()}

    dev_dataset_for_pred = DevDataset(dev_texts, tokenizer, max_length)

    if len(dev_dataset_for_pred) == 0:
        pred_labels = np.array([], dtype=int)
    else:
        pred_out = trainer.predict(dev_dataset_for_pred)
        raw = pred_out.predictions
        if raw is None:
            logits = np.empty((len(dev_dataset_for_pred), 0), dtype=np.float32)
        else:
            if isinstance(raw, (tuple, list)):
                raw = raw[0]
            logits = np.array(raw)

        if logits.ndim == 2 and logits.shape[1] >= 2:
            pred_labels = np.argmax(logits, axis=1).astype(int)
        elif logits.ndim == 2 and logits.shape[1] == 1:
            pred_labels = (logits[:, 0] >= 0).astype(int)
        elif logits.ndim == 1 and logits.shape[0] == len(dev_dataset_for_pred):
            pred_labels = (logits >= 0).astype(int)
        else:
            pred_labels = np.zeros(len(dev_dataset_for_pred), dtype=int)

    pd.DataFrame({"id": dev_ids, "prediction": pred_labels}).to_csv(
        os.path.join(out_dir, "predictions_dev.csv"), index=False
    )

    # Free memory (important on MPS)
    del trainer, model, tokenizer, train_dataset, eval_dataset, dev_dataset_for_pred
    gc.collect()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    return metrics


# CLI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="subtask1")
    parser.add_argument("--train_dir", type=str, default="train")
    parser.add_argument("--dev_dir", type=str, default="dev")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_bs", type=int, default=64)
    parser.add_argument("--eval_bs", type=int, default=8)
    parser.add_argument("--val_from_train", type=float, default=0.1)  # used for held-out split
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--langs", type=str, nargs="*", default=LANGS)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    summary = []
    for lang in args.langs:
        print(f"\n===== Language: {lang} =====")
        try:
            m = run_one_language(
                lang=lang,
                data_root=args.data_root,
                train_dir=args.train_dir,
                dev_dir=args.dev_dir,
                epochs=args.epochs,
                lr=args.lr,
                train_bs=args.train_bs,
                eval_bs=args.eval_bs,
                val_from_train=args.val_from_train,
                seed=args.seed,
                max_length=args.max_length,
            )
            summary.append(m)
        except Exception as e:
            warnings.warn(f"[{lang}] failed: {e}")
            summary.append({
                "lang": lang, "f1_macro": np.nan, "precision_macro": np.nan,
                "recall_macro": np.nan, "used_dev_labels": False,
                "val_from_train": args.val_from_train, "model_name": "ERROR"
            })

    pd.DataFrame(summary).to_csv("outputs/summary_metrics.csv", index=False)
    print("\n=== Summary ===")
    try:
        print(pd.DataFrame(summary)[["lang","f1_macro","precision_macro","recall_macro","used_dev_labels","model_name"]])
    except Exception:
        print(pd.DataFrame(summary))

if __name__ == "__main__":

    main()
