"""
GLiNER Fine-tuning Script — Australian PII Extension
======================================================
Fine-tunes knowledgator/gliner-pii-large-v1.0 on Australian
organisation and location entities.

Usage:
    pip install gliner torch transformers tqdm

    # Fine-tune
    python finetune_gliner_au.py --train training_data/train.json \
                                  --eval  training_data/eval.json \
                                  --output ./gliner-au-pii-v1

    # Test the trained model
    python finetune_gliner_au.py --test --model ./gliner-au-pii-v1
"""

import json
import argparse
from pathlib import Path


# ─── Training ────────────────────────────────────────────────────────────────

def train(train_path: str, eval_path: str, output_dir: str):
    from gliner import GLiNER
    from gliner.training import Trainer, TrainingArguments

    print("Loading base model: knowledgator/gliner-pii-large-v1.0")
    model = GLiNER.from_pretrained("knowledgator/gliner-pii-large-v1.0")

    with open(train_path) as f:
        train_data = json.load(f)
    with open(eval_path) as f:
        eval_data = json.load(f)

    print(f"Train samples: {len(train_data)}")
    print(f"Eval samples:  {len(eval_data)}")

    training_args = TrainingArguments(
        output_dir=output_dir,

        # ── Learning rate ──────────────────────────────────────────────────
        # Low LR is critical — preserves the base model's existing PII 
        # knowledge while learning new AU entities.
        learning_rate=5e-6,

        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        warmup_ratio=0.1,
        weight_decay=0.01,

        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        logging_steps=50,
        report_to="none",  # Set to "wandb" if you use W&B
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
    )

    print("\nStarting training...")
    trainer.train()

    model.save_pretrained(output_dir)
    print(f"\n✓ Model saved to {output_dir}")


# ─── Evaluation helpers ───────────────────────────────────────────────────────

def compute_metrics(model, eval_data: list[dict], entity_labels: list[str]) -> dict:
    """
    Computes per-label precision, recall, F1 on the eval set.
    Matching is exact span + label.
    """
    from collections import defaultdict

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for sample in eval_data:
        tokens = sample["tokenized_text"]
        text = " ".join(tokens)  # Approximate — good enough for eval

        # Gold spans
        gold = set()
        for start, end, label in sample["ner"]:
            span_text = " ".join(tokens[start:end+1])
            gold.add((span_text.lower(), label))

        # Predicted spans
        predictions = model.predict_entities(text, entity_labels, threshold=0.5)
        pred = set()
        for p in predictions:
            pred.add((p["text"].lower(), p["label"]))

        for item in pred & gold:
            tp[item[1]] += 1
        for item in pred - gold:
            fp[item[1]] += 1
        for item in gold - pred:
            fn[item[1]] += 1

    results = {}
    for label in entity_labels:
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0
        r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        results[label] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}

    return results


def evaluate(model_path: str, eval_path: str):
    from gliner import GLiNER

    print(f"Loading model from {model_path}")
    model = GLiNER.from_pretrained(model_path)

    with open(eval_path) as f:
        eval_data = json.load(f)

    labels = ["AU_ORGANISATION", "AU_GOV_AGENCY", "AU_LOCATION"]
    metrics = compute_metrics(model, eval_data, labels)

    print("\n── Evaluation Results ──────────────────────")
    print(f"{'Label':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("─" * 52)
    for label, m in metrics.items():
        print(f"{label:<20} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}")
    print("─" * 52)

    return metrics


# ─── Quick smoke test ─────────────────────────────────────────────────────────

TEST_SENTENCES = [
    "My experience with Centrelink in Parramatta has been frustrating.",
    "I've worked at Woolworths in Bondi for three years.",
    "The ATO office in Canberra processed my return quickly.",
    "Our team at Atlassian is based in Sydney.",
    "Services Australia in Geelong was very helpful with my claim.",
    "I recently moved to Fitzroy and started working at NAB.",
    "The Department of Health office in Darwin needs better staffing.",
    "My employer, BHP, transferred me to their Perth operations.",
    "I contacted the Fair Work Commission about a workplace issue in Melbourne.",
    "CSL's facility in Broadmeadows is a fantastic place to work.",
]

def run_test(model_path: str):
    from gliner import GLiNER

    print(f"Loading model from {model_path}")
    model = GLiNER.from_pretrained(model_path)
    labels = ["AU_ORGANISATION", "AU_GOV_AGENCY", "AU_LOCATION"]

    print("\n── Smoke Test ──────────────────────────────")
    for sentence in TEST_SENTENCES:
        entities = model.predict_entities(sentence, labels, threshold=0.5)
        print(f"\n  Text: {sentence}")
        if entities:
            for e in entities:
                print(f"    [{e['label']}] '{e['text']}' (score: {e['score']:.2f})")
        else:
            print("    (no entities detected)")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GLiNER on Australian entities")

    parser.add_argument("--train",  type=str, default="training_data/train.json")
    parser.add_argument("--eval",   type=str, default="training_data/eval.json")
    parser.add_argument("--output", type=str, default="./gliner-au-pii-v1")

    parser.add_argument("--test",   action="store_true",
                        help="Run smoke test on a trained model (requires --model)")
    parser.add_argument("--model",  type=str,
                        help="Path to trained model for --test or --evaluate")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run full evaluation on eval set (requires --model and --eval)")

    args = parser.parse_args()

    if args.test:
        model_path = args.model or args.output
        run_test(model_path)
    elif args.evaluate:
        model_path = args.model or args.output
        evaluate(model_path, args.eval)
    else:
        train(args.train, args.eval, args.output)
