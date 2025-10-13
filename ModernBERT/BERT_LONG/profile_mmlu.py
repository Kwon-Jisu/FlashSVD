#!/usr/bin/env python3
"""
Profile ModernBERT on MMLU-style inputs with long contexts.

- Loads MMLU (lukaemon/mmlu preferred; falls back to hendrycks_test)
- Packs multiple QA items into long sequences up to --seq-len
- Runs the HF ModernBERT sequence classification model (SDPA backend)
- Reports peak GPU memory and latency; accuracy is not computed (head is SST-2)

Usage examples:
  python profile_imdb.py --model-dir /home/zs89/FlashSVD/ModernBERT/model/modernbert-base-imdb --seq-len 2048 --batch-size 16 --max-batches 50

  python profile_imdb.py --model-dir /home/zs89/FlashSVD/ModernBERT/model/modernbert-base-imdb --seq-len 4096 --batch-size 8 --max-batches 50
"""

import os
import time
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset, concatenate_datasets
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _mmlu_example_to_text(ex):
    # Try common field layouts across MMLU variants
    if "question" in ex:
        q = ex["question"]
    elif "input" in ex:
        q = ex["input"]
    else:
        q = str(ex)

    # Choices
    if "choices" in ex and isinstance(ex["choices"], (list, tuple)) and len(ex["choices"]) >= 4:
        A, B, C, D = ex["choices"][:4]
    else:
        A = ex.get("A") or ex.get("a") or ""
        B = ex.get("B") or ex.get("b") or ""
        C = ex.get("C") or ex.get("c") or ""
        D = ex.get("D") or ex.get("d") or ""

    return f"Q: {q}\nA) {A}\nB) {B}\nC) {C}\nD) {D}\n"


def load_mmlu(split: str = "dev", subjects: Optional[List[str]] = None) -> HFDataset:
    """Load MMLU dataset split, optionally filtered to selected subjects.
    Prefers lukaemon/mmlu; falls back to hendrycks_test.
    Returns a single huggingface Dataset.
    """
    subjects = subjects or []

    # Try lukaemon/mmlu (has configs per subject and 'all')
    try:
        cfgs = subjects if subjects else ["all"]
        ds_list = []
        for cfg in cfgs:
            # Map split alias
            hf_split = {"validation": "dev", "val": "dev"}.get(split, split)
            cur = load_dataset("lukaemon/mmlu", cfg, split=hf_split)
            ds_list.append(cur)
        return concatenate_datasets(ds_list) if len(ds_list) > 1 else ds_list[0]
    except Exception:
        pass

    # Fallback: hendrycks_test (many subject configs; no 'all')
    try:
        cfgs = subjects or [
            # A compact default subject subset if none specified
            "abstract_algebra", "anatomy", "astronomy", "high_school_physics",
            "college_chemistry", "college_mathematics",
        ]
        ds_list = []
        for cfg in cfgs:
            cur = load_dataset("hendrycks_test", cfg, split="validation" if split in ("dev", "validation", "val") else "test")
            ds_list.append(cur)
        return concatenate_datasets(ds_list) if len(ds_list) > 1 else ds_list[0]
    except Exception as e:
        raise RuntimeError(f"Failed to load MMLU from both lukaemon/mmlu and hendrycks_test: {e}")


class PackedMMLUDataset(Dataset):
    """Packs multiple MMLU QA items into long sequences near target seq_len.

    Each item is a dict with tensor fields: input_ids, attention_mask.
    Labels are omitted (profiling only).
    """

    def __init__(
        self,
        hf_ds: HFDataset,
        tokenizer,
        seq_len: int,
        max_packs: int,
    ):
        self.items: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._build(hf_ds, tokenizer, seq_len, max_packs)

    def _build(self, hf_ds: HFDataset, tokenizer, seq_len: int, max_packs: int):
        acc_text = ""
        packs = 0

        def flush_pack(text: str):
            nonlocal packs
            if not text:
                return
            enc = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=seq_len,
                return_tensors="pt",
            )
            self.items.append((enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)))
            packs += 1

        for ex in hf_ds:
            snippet = _mmlu_example_to_text(ex)
            # If current buffer empty, seed it
            cur = snippet if not acc_text else (acc_text + "\n" + snippet)
            # Check tokenized length; if over, flush previous buffer
            enc_len = len(tokenizer(cur, truncation=False)["input_ids"])  # no padding here
            if enc_len >= seq_len:
                if acc_text:
                    flush_pack(acc_text)
                else:
                    # Single example already exceeds; flush snippet directly (will be truncated)
                    flush_pack(snippet)
                acc_text = ""
                if packs >= max_packs:
                    break
            else:
                acc_text = cur

            if packs >= max_packs:
                break

        # Final tail
        if packs < max_packs and acc_text:
            flush_pack(acc_text)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ids, mask = self.items[idx]
        return {"input_ids": ids, "attention_mask": mask}


@torch.no_grad()
def profile_peak_time(model, loader, device):
    # Clean memory and reset peak tracking for accurate measurement
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    steps = 0
    tokens = 0
    start = time.perf_counter()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        tokens += int(batch["attention_mask"].sum().item())
        steps += 1
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed_s = max(1e-9, time.perf_counter() - start)

    # Peak memory during inference
    if torch.cuda.is_available() and device.startswith("cuda"):
        peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        peak_mib = 0.0

    return {
        "steps": steps,
        "elapsed_s": elapsed_s,
        "ms_per_batch": (elapsed_s * 1000.0 / max(1, steps)),
        "tokens": tokens,
        "tok_per_s": (tokens / elapsed_s) if tokens else 0.0,
        "peak_mib": peak_mib,
    }


def main():
    import argparse
    p = argparse.ArgumentParser("ModernBERT MMLU long-context profiler")
    p.add_argument("--model-dir", default="../model/modernbert-base-sst2")
    p.add_argument("--subjects", default="", help="Comma-separated subject list; empty means all")
    p.add_argument("--split", default="val", choices=["dev", "validation", "val", "test"])
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=50, help="Max batches to build (packing units)")
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Model + tokenizer
    cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"  # keep parity with our long-context wrappers
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, config=cfg, trust_remote_code=True
    ).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # Measure persistent allocation baseline
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        persistent_baseline = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Persistent model storage: {persistent_baseline:6.1f} MiB")

    # Data
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    print(f"Loading MMLU split={args.split} subjects={'all' if not subjects else ','.join(subjects)} ...")
    hf_ds = load_mmlu(args.split, subjects)
    print(f"Loaded {len(hf_ds)} raw MMLU examples")

    max_packs = max(1, args.max_batches * args.batch_size)
    packed = PackedMMLUDataset(hf_ds, tok, seq_len=args.seq_len, max_packs=max_packs)
    print(f"Built {len(packed)} packed sequences @ seq_len={args.seq_len}")

    loader = DataLoader(
        packed,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # Profile
    stats = profile_peak_time(model, loader, device)

    # Report
    print(
        f"MMLU profile | steps={stats['steps']} | bs={args.batch_size} | L={args.seq_len} | "
        f"{stats['ms_per_batch']:6.1f} ms/b | {stats['tok_per_s']/1e6:5.2f}M tok/s | peak={stats['peak_mib']:6.1f} MiB"
    )


if __name__ == "__main__":
    main()

