"""
AU GLiNER Training Data Generator
===================================
Generates synthetic Australian NER training data for GLiNER fine-tuning,
targeting AU_ORGANISATION, AU_GOV_AGENCY, and AU_LOCATION entities.

Usage:
    pip install openai tqdm
    export OPENAI_API_KEY=your_key

    # Quick test (50 samples)
    python generate_au_pii_training_data.py --samples 50 --output test_data

    # Full run (5000 samples)
    python generate_au_pii_training_data.py --samples 5000 --output training_data

    # Resume interrupted run
    python generate_au_pii_training_data.py --samples 5000 --output training_data --resume
"""

import json
import random
import time
import argparse
import re
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv
load_dotenv()

# ─── Load seed data ──────────────────────────────────────────────────────────

def load_list(filepath: str) -> list[str]:
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip() and line != '' and not line.startswith("#")]

SUBURBS    = load_list("data/au_suburbs.txt")
COMPANIES  = load_list("data/au_companies.txt")
AGENCIES   = load_list("data/au_gov_agencies.txt")
STATES     = load_list("data/au_states.txt")

# Combine locations (suburbs + states)
LOCATIONS  = SUBURBS + STATES

# ─── Survey contexts ──────────────────────────────────────────────────────────

SURVEY_CONTEXTS = [
    "employee satisfaction survey at an Australian workplace",
    "healthcare or hospital patient experience survey",
    "government services feedback form",
    "community services or NDIS participant survey",
    "university or TAFE student experience survey",
    "housing or rental assistance survey",
    "aged care or disability services feedback",
    "retail or customer service experience survey",
    "transport or infrastructure feedback survey",
    "small business owner feedback survey",
    "job seeker or employment services survey",
    "mental health or wellbeing services feedback",
    "trust and distrust of institutions survey",
    "survey about life in Australian cities and suburbs"
]

# ─── Prompt templates ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You generate realistic Australian survey response sentences for NER (named entity recognition) training data.

Output ONLY a valid JSON object. No explanation, no markdown, no code fences.

Required format:
{
  "samples": [
    {
      "text": "The full response as a string.",
      "entities": [
        {"text": "entity string", "label": "LABEL", "start": N, "end": N}
      ]
    }
  ]
}

Labels to use:
  AU_ORGANISATION  — private companies, banks, retailers, media, tech firms
  AU_GOV_AGENCY    — government departments, regulatory bodies, public services
  AU_LOCATION      — suburbs, cities, towns, states, territories

Hard rules:
  1. Character offsets (start, end) must be EXACT. text[start:end] must equal entity "text".
  2. end = start + len(entity text). Double-check this before outputting.
  3. Use varied sentence structures. Avoid always starting with "I work at...". Some should be responses that are phrase answers with no context.
  4. Entities should appear naturally in realistic survey language.
  5. Mix entity types within responses where natural.
  6. Include a range of response lengths (short, medium, and occasionally longer).
  7. Only use organisations and locations from the provided lists.
  8. Do not invent or hallucinate entity names not in the provided lists.
  9. Make each response mirror real Australian survey responses, with a variety of contexts and sentiments.
10. Do not start any response with {"Living in", "As a", "I recently", "I appreciate", "I often", "The support", "Working with"}
"""

def build_user_prompt(orgs: list, gov: list, locs: list, context: str, n: int) -> str:
    return f"""Generate exactly {n} survey response sentences for this context: "{context}"

Use ONLY these entities (mix them naturally across the {n} sentences):

Organisations (AU_ORGANISATION): {', '.join(orgs)}
Government agencies (AU_GOV_AGENCY): {', '.join(gov)}
Locations (AU_LOCATION): {', '.join(locs)}

Each sentence must contain at least one entity. Aim for 1-3 entities per sentence.
Vary which entity types appear — not every sentence needs all three types."""


# ─── Offset validation & repair ──────────────────────────────────────────────

def validate_and_repair_entities(sample: dict) -> dict | None:
    """
    Validates character offsets for each entity.
    Attempts to repair wrong offsets by searching for the entity text.
    Returns None if the sample is unrecoverable.
    """
    text = sample["text"]
    valid_entities = []

    for ent in sample.get("entities", []):
        ent_text = ent.get("text", "")
        start = ent.get("start")
        end = ent.get("end")

        if not ent_text:
            continue

        # Check if offset is already correct
        if start is not None and end is not None and text[start:end] == ent_text:
            valid_entities.append(ent)
            continue

        # Try to repair: find first occurrence
        idx = text.find(ent_text)
        if idx != -1:
            valid_entities.append({
                "text": ent_text,
                "label": ent["label"],
                "start": idx,
                "end": idx + len(ent_text),
            })
        # else: discard this entity

    if not valid_entities:
        return None  # No valid entities — discard sample

    return {"text": text, "entities": valid_entities}


# ─── Convert to GLiNER tokenized format ──────────────────────────────────────

def simple_tokenize(text: str) -> list[tuple[str, int]]:
    """Returns list of (token, char_start) using whitespace + punctuation split."""
    tokens = []
    for match in re.finditer(r"\S+", text):
        tokens.append((match.group(), match.start()))
    return tokens


def to_gliner_format(sample: dict) -> dict | None:
    """
    Converts a validated sample (char offsets) to GLiNER token-index format.
    Returns None if any entity can't be aligned to token boundaries.
    """
    text = sample["text"]
    token_list = simple_tokenize(text)
    tokens = [t for t, _ in token_list]
    token_starts = [s for _, s in token_list]
    token_ends = [s + len(t) for t, s in token_list]

    ner = []
    for ent in sample["entities"]:
        e_start, e_end = ent["start"], ent["end"]

        # Find token that starts at or contains e_start
        start_tok = next(
            (i for i, ts in enumerate(token_starts) if ts <= e_start < token_ends[i]),
            None
        )
        # Find token that ends at or contains e_end
        end_tok = next(
            (i for i, te in enumerate(token_ends) if token_starts[i] < e_end <= te),
            None
        )

        if start_tok is None or end_tok is None or start_tok > end_tok:
            return None  # Can't align — discard whole sample

        ner.append([start_tok, end_tok, ent["label"]])

    return {"tokenized_text": tokens, "ner": ner}


# ─── API call with retry ──────────────────────────────────────────────────────

def call_gpt(client: OpenAI, orgs, gov, locs, context, n=8, retries=3) -> list[dict]:
    prompt = build_user_prompt(orgs, gov, locs, context, n)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                #max_completion_tokens=2000,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            print(response.choices[0].message.content[:200])  # Print start of response for debugging
            raw = (response.choices[0].message.content or "").strip()

            # Strip accidental markdown fences
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            data = json.loads(raw)
            return data.get("samples", [])

        except json.JSONDecodeError as e:
            print(f"  [JSON error attempt {attempt+1}]: {e}")
            print(f"  Response was: {raw[:200]}...")
            time.sleep(2 ** attempt)
        except RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"  [Rate limit] waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"  [Error attempt {attempt+1}]: {e}")
            time.sleep(2 ** attempt)

    return []


# ─── Main generation loop ─────────────────────────────────────────────────────

def generate(target_samples: int, output_dir: str, resume: bool = False):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_file = output_path / "raw_samples.jsonl"
    gliner_file = output_path / "gliner_samples.jsonl"

    # Resume: count existing samples
    existing = 0
    if resume and raw_file.exists():
        with open(raw_file) as f:
            existing = sum(1 for _ in f)
        print(f"Resuming from {existing} existing samples...")

    client = OpenAI()

    BATCH_SIZE = 8       # Samples per API call
    batches_needed = (target_samples - existing + BATCH_SIZE - 1) // BATCH_SIZE

    raw_out = open(raw_file, "a")
    gliner_out = open(gliner_file, "a")

    collected = existing
    discarded = 0

    with tqdm(total=target_samples, initial=existing, desc="Generating samples") as pbar:
        for _ in range(batches_needed):
            if collected >= target_samples:
                break

            # Sample entity subsets for this batch
            orgs = random.sample(COMPANIES, min(4, len(COMPANIES)))
            gov  = random.sample(AGENCIES, min(3, len(AGENCIES)))
            locs = random.sample(LOCATIONS, min(5, len(LOCATIONS)))
            context = random.choice(SURVEY_CONTEXTS)

            raw_samples = call_gpt(client, orgs, gov, locs, context, n=BATCH_SIZE)

            for raw in raw_samples:
                validated = validate_and_repair_entities(raw)
                if validated is None:
                    discarded += 1
                    continue

                gliner = to_gliner_format(validated)
                if gliner is None:
                    discarded += 1
                    continue

                raw_out.write(json.dumps(validated) + "\n")
                gliner_out.write(json.dumps(gliner) + "\n")
                collected += 1
                pbar.update(1)

                if collected >= target_samples:
                    break

            time.sleep(0.3)  # Polite rate limiting

    raw_out.close()
    gliner_out.close()

    print(f"\n✓ Generated {collected} valid samples ({discarded} discarded)")
    print(f"  Raw samples:    {raw_file}")
    print(f"  GLiNER format:  {gliner_file}")

    # Split into train/eval
    split_train_eval(gliner_file, output_path)


def split_train_eval(gliner_file: Path, output_dir: Path, eval_ratio: float = 0.1):
    with open(gliner_file) as f:
        all_samples = [json.loads(line) for line in f]

    random.shuffle(all_samples)
    split = int(len(all_samples) * (1 - eval_ratio))
    train = all_samples[:split]
    eval_ = all_samples[split:]

    with open(output_dir / "train.json", "w") as f:
        json.dump(train, f, indent=2)
    with open(output_dir / "eval.json", "w") as f:
        json.dump(eval_, f, indent=2)

    print(f"  train.json:     {len(train)} samples")
    print(f"  eval.json:      {len(eval_)} samples")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AU GLiNER training data")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of training samples to generate (default: 500)")
    parser.add_argument("--output", type=str, default="training_data",
                        help="Output directory (default: training_data)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted run")
    args = parser.parse_args()

    generate(args.samples, args.output, args.resume)
