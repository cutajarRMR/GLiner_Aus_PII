"""
GLiNER Fine-tuning Script — Australian PII Extension
======================================================
Tested against GLiNER 0.2.26.

Setup:
    pip install "gliner==0.2.26" torch transformers accelerate tqdm

Fine-tune:
    python finetune_gliner_au.py
    python finetune_gliner_au.py --train training_data/train.json \\
                                  --eval  training_data/eval.json \\
                                  --output ./gliner-au-pii-v1

Smoke test:
    python finetune_gliner_au.py --test --model ./gliner-au-pii-v1

Full eval:
    python finetune_gliner_au.py --evaluate --model ./gliner-au-pii-v1
"""

import json
import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"


# ─── Collator resolution ─────────────────────────────────────────────────────

def get_data_collator(model):
    """
    GLiNER 0.2.26 uses architecture-specific collator classes.
    gliner-pii-large-v1.0 is a span-based UniEncoder model → SpanDataCollator.

    Falls back through known class names so the script survives future renames.
    """
    from gliner.data_processing import collator as collator_module

    # GLiNER 0.2.26 — span-based models
    for cls_name in ("SpanDataCollator", "UniEncoderSpanDataCollator"):
        cls = getattr(collator_module, cls_name, None)
        if cls is not None:
            try:
                return cls(model.config, data_processor=model.data_processor, prepare_labels=True)
            except TypeError:
                try:
                    return cls(model.config)
                except Exception:
                    pass

    # Older GLiNER (pre-0.2.x) — generic collators
    for cls_name in ("DataCollator", "DataCollatorWithPadding"):
        cls = getattr(collator_module, cls_name, None)
        if cls is not None:
            try:
                return cls(model.config, data_processor=model.data_processor, prepare_labels=True)
            except TypeError:
                try:
                    return cls(model.config)
                except Exception:
                    pass

    # Nothing worked — print what IS available to help debug
    available = [x for x in dir(collator_module) if not x.startswith("_") and x[0].isupper()]
    print(f"\n  Could not resolve a collator. Available classes: {available}")
    print("  Falling back to Trainer default — training may fail.\n")
    return None


# ─── Training ────────────────────────────────────────────────────────────────

def train(train_path: str, eval_path: str, output_dir: str):
    import torch
    from gliner import GLiNER
    from gliner.training import Trainer, TrainingArguments

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("  Warning: no GPU detected — training will be very slow.")

    print("Loading base model: knowledgator/gliner-pii-large-v1.0")
    model = GLiNER.from_pretrained("knowledgator/gliner-pii-large-v1.0")
    model.to(device)

    with open(train_path) as f:
        train_data = json.load(f)
    with open(eval_path) as f:
        eval_data = json.load(f)

    print(f"Train samples : {len(train_data)}")
    print(f"Eval samples  : {len(eval_data)}")

    data_collator = get_data_collator(model)
    print(f"Collator      : {type(data_collator).__name__ if data_collator else 'Trainer default'}")

    training_args = TrainingArguments(
        output_dir=output_dir,

        # Low LR preserves existing PII knowledge while learning AU entities
        learning_rate=5e-6,

        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        warmup_ratio=0.1,
        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        logging_steps=50,
        report_to="none",
        seed=42,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
    )
    if data_collator is not None:
        trainer_kwargs["data_collator"] = data_collator

    trainer = Trainer(**trainer_kwargs)

    print("\nStarting training...")
    trainer.train()

    model.save_pretrained(output_dir)
    print(f"\n✓ Model saved to {output_dir}")


# ─── Evaluation ──────────────────────────────────────────────────────────────

def compute_metrics(model, eval_data: list, entity_labels: list) -> dict:
    from collections import defaultdict

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for sample in eval_data:
        tokens = sample["tokenized_text"]
        text = " ".join(tokens)

        gold = set()
        for start, end, label in sample["ner"]:
            span_text = " ".join(tokens[start:end + 1])
            gold.add((span_text.lower(), label))

        predictions = model.predict_entities(text, entity_labels, threshold=0.5)
        pred = {(p["text"].lower(), p["label"]) for p in predictions}

        for item in pred & gold:
            tp[item[1]] += 1
        for item in pred - gold:
            fp[item[1]] += 1
        for item in gold - pred:
            fn[item[1]] += 1

    results = {}
    for label in entity_labels:
        p  = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0
        r  = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        results[label] = {
            "precision": round(p,  3),
            "recall":    round(r,  3),
            "f1":        round(f1, 3),
        }
    return results


def evaluate(model_path: str, eval_path: str):
    from gliner import GLiNER

    print(f"Loading model from {model_path}")
    model = GLiNER.from_pretrained(model_path)

    with open(eval_path) as f:
        eval_data = json.load(f)

    labels  = ["AU_ORGANISATION", "AU_GOV_AGENCY", "AU_LOCATION"]
    metrics = compute_metrics(model, eval_data, labels)

    print("\n── Evaluation Results ──────────────────────────")
    print(f"{'Label':<22} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("─" * 54)
    for label, m in metrics.items():
        print(f"{label:<22} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}")
    print("─" * 54)

    return metrics


# ─── Smoke test ───────────────────────────────────────────────────────────────

TEST_SENTENCES = [
    # Well-known — base model should handle these already
    "My experience with Centrelink in Parramatta has been frustrating.",
    "The ATO office in Canberra processed my return quickly.",
    "I contacted the Fair Work Commission about a workplace issue in Melbourne.",
    # AU-specific — what the fine-tuning is for
    "Three calls to TfNSW and still no response about my Opal card.",
    "I moved to Nhulunbuy last year and struggle to access Lumus Imaging.",
    "The SA Department for Education office in Mitcham was very helpful.",
    "Guzman y Gomez in Indooroopilly — great service, friendly staff.",
    "My TAFE Queensland course in Bundaberg starts next month.",
    "After relocating to Warakurna, accessing Horizon Power has been difficult.",
    "Slater and Gordon handled my WorkCover claim in Geelong.",
    # Short phrase answers
    "Centrelink, Dubbo.",
    "Worked at Bapcor in Caringbah for 5 years.",
    # Terse / complaint tone
    "Three weeks waiting for SA Health to respond.",
    # Ambiguous single-word location
    "The Eden office never picks up the phone.",
]


def run_test(model_path: str):
    from gliner import GLiNER

    print(f"Loading model from {model_path}")
    model  = GLiNER.from_pretrained(model_path)
    labels = ["AU_ORGANISATION", "AU_GOV_AGENCY", "AU_LOCATION"]

    print("\n── Smoke Test ────────────────────────────────────")
    for sentence in TEST_SENTENCES:
        entities = model.predict_entities(sentence, labels, threshold=0.5)
        print(f"\n  Text: {sentence}")
        if entities:
            for e in entities:
                print(f"    [{e['label']}] '{e['text']}' ({e['score']:.2f})")
        else:
            print("    (no entities detected)")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GLiNER on Australian entities")

    parser.add_argument("--train",    type=str, default="training_data/train.json")
    parser.add_argument("--eval",     type=str, default="training_data/eval.json")
    parser.add_argument("--output",   type=str, default="./gliner-au-pii-v1")
    parser.add_argument("--test",     action="store_true",
                        help="Smoke test a trained model")
    parser.add_argument("--evaluate", action="store_true",
                        help="Full precision/recall/F1 on eval set")
    parser.add_argument("--model",    type=str,
                        help="Path to trained model (for --test or --evaluate)")

    args = parser.parse_args()

    if args.test:
        run_test(args.model or args.output)
    elif args.evaluate:
        evaluate(args.model or args.output, args.eval)
    else:
        train(args.train, args.eval, args.output)