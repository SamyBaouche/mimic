"""Element matcher: a small Hugging Face cross-encoder fine-tuned to re-find UI elements.

Given the element the user used in the demo and one candidate on the redesigned page, it
scores "same purpose?". It replaces most calls to the 7B vision model: milliseconds on a CPU
instead of seconds on a GPU. Qwen2.5-VL is only asked when the matcher is unsure.

    uv sync --extra matcher
    uv run python -m mimic.matcher train          # fine-tune + compare with baselines
    uv run python -m mimic.matcher evaluate       # re-run the evaluation of a saved model
"""

from __future__ import annotations

import argparse
import json
import random
import time
from difflib import SequenceMatcher
from pathlib import Path

from mimic.matcher_data import Example, make_examples, sandbox_v2_examples, to_pairs

DEFAULT_BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODEL_DIR = Path("models/element_matcher")


# ---------------------------------------------------------------- inference


class ElementMatcher:
    """Scores (demo element, candidate) pairs with a sequence-classification model."""

    def __init__(self, model, tokenizer, threshold: float = 0.5, max_length: int = 64):
        import torch

        self.torch = torch
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.max_length = max_length

    @classmethod
    def load(cls, path: Path = MODEL_DIR) -> ElementMatcher:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        meta_path = Path(path) / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        model = AutoModelForSequenceClassification.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        return cls(model, tokenizer, threshold=meta.get("threshold", 0.5))

    def scores(self, target: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        batch = self.tokenizer(
            [target] * len(candidates),
            candidates,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with self.torch.no_grad():
            logits = self.model(**batch).logits.squeeze(-1)
        return self.torch.sigmoid(logits).reshape(-1).tolist()


# ---------------------------------------------------------------- evaluation


def top1_accuracy(score_fn, examples: list[Example]) -> float:
    """Share of redesigns where the best-scored candidate is the right one."""
    hits = 0
    for ex in examples:
        scores = score_fn(ex.target.text(), [c.text() for c in ex.candidates])
        hits += int(max(range(len(scores)), key=scores.__getitem__) == ex.answer)
    return hits / max(len(examples), 1)


def string_similarity(target: str, candidates: list[str]) -> list[float]:
    """Baseline: how alike the texts look, letter by letter."""

    def label(t):
        return t.split("'")[1].lower() if "'" in t else t.lower()

    return [SequenceMatcher(None, label(target), label(c)).ratio() for c in candidates]


def latency_ms(matcher: ElementMatcher, examples: list[Example], n: int = 50) -> float:
    sample = examples[:n]
    t0 = time.perf_counter()
    for ex in sample:
        matcher.scores(ex.target.text(), [c.text() for c in ex.candidates])
    return (time.perf_counter() - t0) * 1000 / max(len(sample), 1)


def evaluate(matcher: ElementMatcher, test: list[Example], baseline_scores: dict) -> dict:
    sandbox = sandbox_v2_examples()
    results = {
        "fine_tuned": {
            "synthetic_top1": round(top1_accuracy(matcher.scores, test), 4),
            "sandbox_v2_top1": round(top1_accuracy(matcher.scores, sandbox), 4),
            "ms_per_decision_cpu": round(latency_ms(matcher, test), 1),
        },
        "string_similarity": {
            "synthetic_top1": round(top1_accuracy(string_similarity, test), 4),
            "sandbox_v2_top1": round(top1_accuracy(string_similarity, sandbox), 4),
        },
    }
    results.update(baseline_scores)
    return results


# ---------------------------------------------------------------- training


def train(
    model,
    tokenizer,
    train_examples: list[Example],
    test_examples: list[Example],
    epochs: int = 2,
    lr: float = 2e-5,
    batch_size: int = 32,
    seed: int = 0,
    out: Path | None = MODEL_DIR,
    base_name: str = "custom",
    log=print,
) -> dict:
    import torch

    torch.manual_seed(seed)
    rng = random.Random(seed)

    baselines = {}
    zero_shot = ElementMatcher(model, tokenizer)
    baselines["before_fine_tuning"] = {
        "synthetic_top1": round(top1_accuracy(zero_shot.scores, test_examples), 4),
        "sandbox_v2_top1": round(top1_accuracy(zero_shot.scores, sandbox_v2_examples()), 4),
    }

    pairs = to_pairs(train_examples)
    positives = sum(p[2] for p in pairs)
    pos_weight = torch.tensor((len(pairs) - positives) / max(positives, 1))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    t0 = time.perf_counter()
    losses = []
    for epoch in range(epochs):
        rng.shuffle(pairs)
        total = 0.0
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            batch = tokenizer(
                [a for a, _, _ in chunk],
                [b for _, b, _ in chunk],
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            labels = torch.tensor([y for _, _, y in chunk])
            logits = model(**batch).logits.squeeze(-1).reshape(-1)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(chunk)
        losses.append(round(total / len(pairs), 4))
        log(f"  epoch {epoch + 1}/{epochs}  loss {losses[-1]:.4f}")
    train_seconds = time.perf_counter() - t0

    matcher = ElementMatcher(model, tokenizer)
    meta = {
        "base_model": base_name,
        "threshold": 0.5,
        "train_examples": len(train_examples),
        "train_pairs": len(pairs),
        "test_examples": len(test_examples),
        "epochs": epochs,
        "loss_per_epoch": losses,
        "train_seconds": round(train_seconds, 1),
        "parameters": sum(p.numel() for p in model.parameters()),
        "results": evaluate(matcher, test_examples, baselines),
        "held_out_note": "LedgerLite v2 labels never appear in training (see matcher_data.py)",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def print_results(meta: dict) -> None:
    print(
        f"\nBase: {meta['base_model']}  ({meta['parameters'] / 1e6:.1f}M parameters)  "
        f"trained on {meta['train_pairs']} pairs in {meta['train_seconds']} s"
    )
    print(f"\n{'model':<22}{'synthetic top-1':>17}{'LedgerLite v2 top-1':>22}")
    for name, r in meta["results"].items():
        print(f"{name:<22}{r['synthetic_top1']:>16.1%}{r['sandbox_v2_top1']:>22.1%}")
    ms = meta["results"]["fine_tuned"]["ms_per_decision_cpu"]
    print(f"\nFine-tuned matcher: {ms} ms per decision on CPU")
    print("LedgerLite v2 labels were held out of training: this is generalisation.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Element matcher (Hugging Face)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--base", default=DEFAULT_BASE, help="Hugging Face model id")
    t.add_argument("--examples", type=int, default=3000)
    t.add_argument("--test", type=int, default=500)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--lr", type=float, default=2e-5)
    sub.add_parser("evaluate")
    args = parser.parse_args(argv)

    if args.cmd == "train":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        print(f"Downloading {args.base} from Hugging Face (first time only)...")
        tokenizer = AutoTokenizer.from_pretrained(args.base)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base, num_labels=1, ignore_mismatched_sizes=True
        )
        train_set = make_examples(args.examples, seed=1, split="train")
        test_set = make_examples(args.test, seed=2, split="test")
        print(f"Training on {len(train_set)} synthetic redesigns...")
        meta = train(
            model, tokenizer, train_set, test_set, args.epochs, args.lr, base_name=args.base
        )
        print_results(meta)
        print(f"\nSaved to {MODEL_DIR}")
    else:
        matcher = ElementMatcher.load()
        meta = json.loads((MODEL_DIR / "meta.json").read_text())
        meta["results"] = evaluate(matcher, make_examples(500, seed=2, split="test"), {}) | {
            k: v for k, v in meta["results"].items() if k != "fine_tuned"
        }
        print_results(meta)


if __name__ == "__main__":
    main()
