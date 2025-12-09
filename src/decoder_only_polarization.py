import os
import json
import argparse
import warnings
import inspect
import gc
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model



LANGS = ["amh", "arb", "deu", "eng", "hau", "ita", "spa", "urd", "zho"]



# Data utilities


def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def clean_df(df: pd.DataFrame, expect_labels: bool) -> pd.DataFrame:
    # id
    if "id" not in df.columns:
        df["id"] = np.arange(len(df))
    # text
    df["text"] = df.get("text", "").fillna("").astype(str)

    if expect_labels:
        if "polarization" not in df.columns:
            raise ValueError("Expected labels but polarization column is missing.")
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



# Prompts


def make_train_prompt(text: str, label: int) -> str:
    """
    Prompt used for training / validation.
    """
    return (
        "You are a polarization classifier.\n"
        "Given a text, output its polarization label (0 = non-polarized, 1 = polarized).\n\n"
        f"Text: {text}\n"
        f"Polarization (0 or 1): {label}"
    )


def make_infer_prompt(text: str) -> str:
    """
    Prompt used for inference (no label).
    """
    return (
        "You are a polarization classifier.\n"
        "Given a text, output its polarization label (0 = non-polarized, 1 = polarized).\n\n"
        f"Text: {text}\n"
        "Polarization (0 or 1):"
    )



# Dataset for causal LM


class CausalLMDataset(Dataset):
    """
    For training / validation.

    We create a full prompt including the label. The labels for the LM are simply
    the input_ids (standard causal LM training: predict next token).
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 256,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]) if self.texts[idx] is not None else ""
        label = int(self.labels[idx])
        prompt = make_train_prompt(text, label)

        enc = self.tokenizer(
            prompt,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # For causal LM, labels are the same as input_ids
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = item["input_ids"].clone()
        return item


class DevCausalLMDataset(Dataset):
    """
    For dev/inference: only text, no labels.
    """

    def __init__(self, texts: List[str], tokenizer, max_length: int = 256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]) if self.texts[idx] is not None else ""
        prompt = make_infer_prompt(text)
        enc = self.tokenizer(
            prompt,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        return item



# TrainingArguments helper


def make_training_args(out_dir, epochs, lr, train_bs, eval_bs, seed, disable_tqdm=False):
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters

    train_bs = max(1, min(train_bs, 8))
    eval_bs = max(1, min(eval_bs, 8))

    kwargs = dict(
        output_dir=out_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        logging_steps=50,
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



# LoRA helpers


def add_lora_to_gpt2(model):
    """
    Attach LoRA adapters to GPT-2 attention + MLP.
    """
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj"],  
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model



# Model choice


def pick_decoder_model_name(lang: str) -> str:
    """
    Use GPT-2 for all languages.
    """
    return "gpt2"



# Inference utilities giving logits to 0/1 label


def predict_labels_logits(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 256,
    threshold: float = 0.5,
) -> List[int]:
    """
    For each text:
      - build the inference prompt,
      - run model forward once,
      - look at last-token logits for tokens "0" and "1",
      - output label 0/1 by probability threshold.
    """
    device = next(model.parameters()).device
    vocab = tokenizer.get_vocab()
    id_0 = vocab.get("0")
    id_1 = vocab.get("1")

    if id_0 is None or id_1 is None:
        raise ValueError("Tokenizer must contain '0' and '1' tokens for classification.")

    preds: List[int] = []
    model.eval()

    for text in texts:
        prompt = make_infer_prompt(text)
        enc = tokenizer(
            prompt,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0, -1, :]  # last token

        two = torch.stack([logits[id_0], logits[id_1]])  # [2]
        probs = torch.softmax(two, dim=-1)
        label = int((probs[1] >= threshold).item())  # 1 if polarized prob >= threshold else 0
        preds.append(label)

    return preds



# One-language runner


def run_one_language_decoder(
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
    max_length: int = 256,
    gen_max_new_tokens: int = 8,
) -> Dict[str, str]:
    train_path = os.path.join(data_root, train_dir, f"{lang}.csv")
    dev_path = os.path.join(data_root, dev_dir, f"{lang}.csv")
    out_dir = os.path.join("outputs_decoder_only", lang)
    os.makedirs(out_dir, exist_ok=True)

    # Load & clean
    train_df = clean_df(safe_read_csv(train_path), expect_labels=True)
    dev_df = clean_df(safe_read_csv(dev_path), expect_labels=False)

    # Split train into train/val
    effective_val = float(val_from_train) if val_from_train and val_from_train > 0 else 0.10
    tr_df, val_df = split_train_val(train_df, effective_val, seed)

    # limit training samples per lang to keep runtime manageable
    MAX_TRAIN_SAMPLES = 400
    if len(tr_df) > MAX_TRAIN_SAMPLES:
        tr_df = tr_df.sample(n=MAX_TRAIN_SAMPLES, random_state=seed).reset_index(drop=True)

    # leakage guard
    train_ids = set(tr_df["id"].tolist())
    val_ids = set(val_df["id"].tolist())
    assert train_ids.isdisjoint(val_ids), "Train/Val overlap detected!"
    with open(os.path.join(out_dir, "split_ids.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "val_from_train": effective_val,
                "train_ids": sorted(list(train_ids)),
                "val_ids": sorted(list(val_ids)),
            },
            f,
        )

    # Tokenizer and model
    model_name = pick_decoder_model_name(lang)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False

    # Adding LoRA
    print(f"Adding LoRA adapters for {lang} ...")
    model = add_lora_to_gpt2(model)

    # Datasets
    train_dataset = CausalLMDataset(
        tr_df["text"].tolist(), tr_df["polarization"].tolist(), tokenizer, max_length=max_length
    )
    eval_dataset = CausalLMDataset(
        val_df["text"].tolist(), val_df["polarization"].tolist(), tokenizer, max_length=max_length
    )

    # Training args
    training_args = make_training_args(
        out_dir=out_dir,
        epochs=epochs,
        lr=lr,
        train_bs=train_bs,
        eval_bs=eval_bs,
        seed=seed,
    )

    def data_collator(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # dynamic padding for causal LM
        batch = {}
        keys = features[0].keys()
        max_len = max(f["input_ids"].shape[0] for f in features)
        for k in keys:
            tensors = []
            for f in features:
                t = f[k]
                pad_len = max_len - t.shape[0]
                if pad_len > 0:
                    pad_val = tokenizer.pad_token_id if k != "labels" else -100
                    t = torch.cat(
                        [t, torch.full((pad_len,), pad_val, dtype=t.dtype)]
                    )
                tensors.append(t)
            batch[k] = torch.stack(tensors)
        return batch

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Log basics
    with open(os.path.join(out_dir, "training_log.txt"), "w", encoding="utf-8") as f:
        f.write(f"Lang: {lang}\nModel: {model_name}+LoRA\n")
        f.write(
            f"epochs={epochs}, lr={lr}, "
            f"train_bs={training_args.per_device_train_batch_size}, "
            f"eval_bs={training_args.per_device_eval_batch_size}, "
            f"val_from_train={effective_val}, seed={seed}\n"
        )

    # Train and eval (loss-based)
    trainer.train()
    eval_results = trainer.evaluate()
    with open(os.path.join(out_dir, "metrics_lm.json"), "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    # ------------------------ VAL METRICS ------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    val_texts = val_df["text"].tolist()
    val_gold = val_df["polarization"].tolist()

    val_preds = predict_labels_logits(
        model, tokenizer, val_texts, max_length=max_length, threshold=0.5
    )

    acc = accuracy_score(val_gold, val_preds)
    macro_f1 = f1_score(val_gold, val_preds, average="macro")
    weighted_f1 = f1_score(val_gold, val_preds, average="weighted")
    support = len(val_gold)

    gold_arr = np.array(val_gold)
    pred_arr = np.array(val_preds)
    total_0 = int((gold_arr == 0).sum())
    total_1 = int((gold_arr == 1).sum())
    correct_0 = int(((gold_arr == 0) & (pred_arr == 0)).sum())
    correct_1 = int(((gold_arr == 1) & (pred_arr == 1)).sum())

    # save per-lang val predictions
    pd.DataFrame(
        {
            "id": val_df["id"],
            "gold": val_gold,
            "pred": val_preds,
        }
    ).to_csv(os.path.join(out_dir, "val_predictions.csv"), index=False)

    # ------------------------ DEV PREDICTIONS  -------------------
    dev_texts = dev_df["text"].tolist()
    dev_ids = dev_df["id"].tolist()

    if len(dev_texts) > 0:
        dev_preds = predict_labels_logits(
            model, tokenizer, dev_texts, max_length=max_length, threshold=0.5
        )
    else:
        dev_preds = []

    pd.DataFrame({"id": dev_ids, "prediction": dev_preds}).to_csv(
        os.path.join(out_dir, "predictions_dev.csv"), index=False
    )

   
    del trainer, model, tokenizer, train_dataset, eval_dataset
    gc.collect()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    return {
        "lang": lang,
        "model_name": "gpt2+LoRA",
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "support": support,
        "correct_0": correct_0,
        "total_0": total_0,
        "correct_1": correct_1,
        "total_1": total_1,
    }


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="subtask1")
    parser.add_argument("--train_dir", type=str, default="train")
    parser.add_argument("--dev_dir", type=str, default="dev")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--train_bs", type=int, default=2)
    parser.add_argument("--eval_bs", type=int, default=2)
    parser.add_argument("--val_from_train", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--langs", type=str, nargs="*", default=LANGS)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--gen_max_new_tokens", type=int, default=8)
    args = parser.parse_args()

    os.makedirs("outputs_decoder_only", exist_ok=True)

    per_lang_rows = []

    for lang in args.langs:
        print(f"\n===== Decoder-only Language (GPT-2 + LoRA): {lang} =====")
        try:
            m = run_one_language_decoder(
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
                gen_max_new_tokens=args.gen_max_new_tokens,
            )
            per_lang_rows.append(m)
        except Exception as e:
            warnings.warn(f"[{lang}] decoder-only run failed: {e}")
            per_lang_rows.append({
                "lang": lang,
                "model_name": "ERROR",
                "accuracy": np.nan,
                "macro_f1": np.nan,
                "weighted_f1": np.nan,
                "support": 0,
                "correct_0": 0,
                "total_0": 0,
                "correct_1": 0,
                "total_1": 0,
            })

    # Saving summary CSV
    pd.DataFrame(per_lang_rows).to_csv("outputs_decoder_only/summary_models.csv", index=False)

    
    print("\n=== PER-LANGUAGE METRICS ===")
    print(f"{'Language':<6} {'Accuracy':<10} {'Macro F1':<10} {'Weighted F1':<12} "
          f"{'Support':<8} {'Class 0':<12} {'Class 1':<12}")
    print("-" * 90)

    total_support = 0
    total_correct_0 = 0
    total_total_0 = 0
    total_correct_1 = 0
    total_total_1 = 0

    for row in per_lang_rows:
        lang = row["lang"]
        acc = row["accuracy"]
        macro_f1 = row["macro_f1"]
        weighted_f1 = row["weighted_f1"]
        support = row["support"]
        c0 = row["correct_0"]
        t0 = row["total_0"]
        c1 = row["correct_1"]
        t1 = row["total_1"]

        total_support += support
        total_correct_0 += c0
        total_total_0 += t0
        total_correct_1 += c1
        total_total_1 += t1

        if support > 0:
            print(f"{lang:<6} {acc:<10.4f} {macro_f1:<10.4f} {weighted_f1:<12.4f} "
                  f"{support:<8d} {f'{c0}/{t0}':<12} {f'{c1}/{t1}':<12}")
        else:
            print(f"{lang:<6} {'nan':<10} {'nan':<10} {'nan':<12} "
                  f"{support:<8d} {'0/0':<12} {'0/0':<12}")

    print("-" * 90)
    print(f"{'TOTAL':<6} {'':<10} {'':<10} {'':<12} {total_support:<8d} "
          f"{f'{total_correct_0}/{total_total_0}':<12} {f'{total_correct_1}/{total_total_1}':<12}")

    print("\n=== Decoder-only Summary (GPT-2 + LoRA) ===")
    print(pd.DataFrame(per_lang_rows)[["lang", "model_name"]])


if __name__ == "__main__":
    main()
