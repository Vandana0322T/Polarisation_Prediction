# src/train_T5.py
import os
import json
import argparse
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)

# --------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------


class PolarisationT5Dataset(Dataset):
    """
    Labeled dataset: text + integer label 0/1.
    This is used for the internal train/val split coming from train/{lang}.csv.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_source_len: int = 128,
        max_target_len: int = 3,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = str(self.texts[idx])
        label_int = int(self.labels[idx])
        target_text = str(label_int)  # "0" or "1"

        # Encode source
        source_enc = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_source_len,
        )

        # Encode target
        with self.tokenizer.as_target_tokenizer():
            target_enc = self.tokenizer(
                target_text,
                truncation=True,
                padding=False,
                max_length=self.max_target_len,
            )

        source_enc["labels"] = target_enc["input_ids"]
        return source_enc


# --------------------------------------------------------------------
# Helpers for loading data
# --------------------------------------------------------------------


def load_train_dataframe(data_root: str, lang: str) -> pd.DataFrame:
    """
    Load train/{lang}.csv which MUST contain 'text' and 'polarization'.
    """
    path = os.path.join(data_root, "train", f"{lang}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find train file: {path}")

    df = pd.read_csv(path)

    if "text" not in df.columns or "polarization" not in df.columns:
        raise ValueError(
            f"{path} must have columns 'text' and 'polarization', got: {df.columns}"
        )

    df = df.dropna(subset=["text", "polarization"]).copy()
    df["polarization"] = pd.to_numeric(df["polarization"], errors="coerce")
    df = df.dropna(subset=["polarization"]).copy()
    df["polarization"] = df["polarization"].astype(int).clip(0, 1)

    return df


def load_dev_dataframe(data_root: str, lang: str) -> pd.DataFrame:
    """
    Load dev/{lang}.csv which contains *only text* (and usually id).
    No labels are used here.
    """
    path = os.path.join(data_root, "dev", f"{lang}.csv")
    if not os.path.exists(path):
        print(f"[{lang}] WARNING: dev file not found: {path} – no dev predictions.")
        return None

    df = pd.read_csv(path)

    if "text" not in df.columns:
        raise ValueError(f"{path} must contain a 'text' column, got: {df.columns}")

    df = df.dropna(subset=["text"]).copy()
    return df


# --------------------------------------------------------------------
# Metrics / decoding
# --------------------------------------------------------------------


def decode_labels_from_ids(
    ids: np.ndarray,
    tokenizer: AutoTokenizer,
) -> List[int]:
    """
    Turn generated token IDs into integer labels 0/1.
    We decode each sequence and look at the first character.
    """
    ids = np.where(ids == -100, tokenizer.pad_token_id, ids)
    texts = tokenizer.batch_decode(ids, skip_special_tokens=True)

    out: List[int] = []
    for t in texts:
        t = str(t).strip()
        if not t:
            out.append(0)
        else:
            ch = t[0]
            if ch in ("0", "1"):
                out.append(int(ch))
            else:
                out.append(0)
    return out


def make_compute_metrics(tokenizer: AutoTokenizer):
    def compute_metrics(eval_pred):
        pred_ids, label_ids = eval_pred
        # predictions and labels are sequences of token IDs
        y_pred = decode_labels_from_ids(pred_ids, tokenizer)

        label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)
        y_true = decode_labels_from_ids(label_ids, tokenizer)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        return {
            "f1_macro": float(f1),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
        }

    return compute_metrics




def generate_labels_for_texts(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_source_len: int,
    max_target_len: int,
    batch_size: int = 16,
) -> List[int]:
    """
    Run T5 generate() on raw texts in small batches and return 0/1 labels.
    This bypasses Seq2SeqTrainer.predict so we never pass bad kwargs
    like `predict_with_generate` into model.generate().
    """
    model.eval()
    device = model.device
    all_preds: List[int] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_source_len,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            gen_ids = model.generate(
                **enc,
                max_length=max_target_len,
            )

            batch_preds = decode_labels_from_ids(
                gen_ids.cpu().numpy(), tokenizer
            )
            all_preds.extend(batch_preds)

    return all_preds


# --------------------------------------------------------------------
# Train + predict for a single language
# --------------------------------------------------------------------


def run_one_language(
    lang: str,
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
) -> Dict[str, Any]:
    print(f"\n===== T5 Language: {lang} =====")

    # 1) Load labeled train data
    train_df = load_train_dataframe(args.data_root, lang)
    print(f"[{lang}] Loaded {len(train_df)} train rows.")

    # 2) Internal train/val split 
    train_df_split, val_df_split = train_test_split(
        train_df,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=train_df["polarization"],
    )
    print(
        f"[{lang}] After split: {len(train_df_split)} train / "
        f"{len(val_df_split)} val rows (val_ratio={args.val_ratio})."
    )

    # 3) Build labeled datasets
    train_dataset = PolarisationT5Dataset(
        train_df_split["text"].tolist(),
        train_df_split["polarization"].tolist(),
        tokenizer,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
    )

    val_dataset = PolarisationT5Dataset(
        val_df_split["text"].tolist(),
        val_df_split["polarization"].tolist(),
        tokenizer,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
    )

    # 4) Load model
    model_name = "t5-small"
    print("Loading T5-small model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Force CPU 
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    output_dir = os.path.join("outputs_t5", f"{lang}_{model_name}")

    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        logging_steps=50,
        predict_with_generate=True,  # for validation metrics only
        load_best_model_at_end=False,
        report_to=[],  # no WandB / TensorBoard
        no_cuda=True,  # CPU only
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding="longest",
        max_length=args.max_source_len,
        label_pad_token_id=-100,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=make_compute_metrics(tokenizer),
    )

    # 5) Train on internal train split
    trainer.train()

    # 6) Evaluate on internal val split
    metrics = trainer.evaluate()
    print(f"[{lang}] validation metrics: {metrics}")

    summary_row: Dict[str, Any] = {
        "lang": lang,
        "f1_macro": float(metrics.get("eval_f1_macro", np.nan)),
        "precision_macro": float(metrics.get("eval_precision_macro", np.nan)),
        "recall_macro": float(metrics.get("eval_recall_macro", np.nan)),
        "model_name": model_name,
        "used_dev_labels": False,  # dev has no labels
    }

    # 7) Predict on dev (text only, no labels) – manual generate()
    dev_df = load_dev_dataframe(args.data_root, lang)
    if dev_df is not None and len(dev_df) > 0:
        dev_texts = dev_df["text"].astype(str).tolist()
        dev_ids = (
            dev_df["id"].tolist()
            if "id" in dev_df.columns
            else list(range(len(dev_texts)))
        )

        try:
            pred_labels = generate_labels_for_texts(
                model=trainer.model,
                tokenizer=tokenizer,
                texts=dev_texts,
                max_source_len=args.max_source_len,
                max_target_len=args.max_target_len,
                batch_size=args.eval_bs,
            )

            out_df = pd.DataFrame(
                {"id": dev_ids, "polarization": pred_labels}
            )
            pred_path = os.path.join(
                "outputs_t5", f"{lang}_dev_predictions_t5.csv"
            )
            out_df.to_csv(pred_path, index=False)
            print(f"[{lang}] dev predictions saved to: {pred_path}")
        except Exception as e:
            print(f"[{lang}] WARNING: dev prediction failed with error: {repr(e)}")

    return summary_row


# --------------------------------------------------------------------
# CLI + main
# --------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "T5-small baseline for polarization detection (binary) "
            "using the BERT baseline pipeline."
        )
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="subtask1",
        help="Root directory with train/ and dev/ subfolders.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=3.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--train_bs",
        type=int,
        default=8,
        help="Train batch size per device.",
    )
    parser.add_argument(
        "--eval_bs",
        type=int,
        default=8,
        help="Eval/dev batch size per device.",
    )
    parser.add_argument(
        "--max_source_len",
        type=int,
        default=128,
        help="Max token length for input text.",
    )
    parser.add_argument(
        "--max_target_len",
        type=int,
        default=3,
        help="Max tokens for target sequence ('0'/'1').",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of train data reserved as validation (like BERT baseline).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["eng"],
        help="Languages to train, e.g. --langs eng amh arb deu ...",
    )

    args = parser.parse_args()
    print("Arguments:", json.dumps(vars(args), indent=2))
    set_seed(args.seed)

    os.makedirs("outputs_t5", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    summary_rows: List[Dict[str, Any]] = []
    for lang in args.langs:
        try:
            row = run_one_language(lang, args, tokenizer)
        except Exception as e:
            print(f"Language {lang} failed with error: {repr(e)}")
            row = {
                "lang": lang,
                "f1_macro": np.nan,
                "precision_macro": np.nan,
                "recall_macro": np.nan,
                "model_name": "t5-small",
                "used_dev_labels": False,
            }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join("outputs_t5", "summary_metrics_t5.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\n=== Overall T5 Summary ===")
    print(summary_df)
    print(f"Overall T5 Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
