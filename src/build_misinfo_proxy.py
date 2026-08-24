"""Build a synthetic Tshivenda misinformation-classification proxy dataset.

Fallback for Objective 4 / proposal §4.5: the Mukwevho et al. (2024) Tshivenda
misinformation dataset is not accessible to this team (confirmed unavailable -
not publicly released, and the authors could not provide it). The proposal
already uses a "news topic classification as downstream proxy task" fallback
for Setswana and Sepedi (§3.2) when no dedicated misinformation dataset
exists; this script builds the equivalent proxy for Tshivenda from the one
labeled-adjacent text resource we do have: the Vukuzenzele government
magazine corpus (dataset/vukuzenzele/), which is all genuine, fact-checked
government text (label = real).

Synthetic "fake" counterparts are generated per real article by applying
content-level distortions that mirror how real misinformation misrepresents
genuine news, not ASR-style noise (that is a separate, already-built system -
see src/corrupt_transcripts.py):

- numeric distortion: swap statistics/percentages/dates for different,
  differently-valued numbers drawn from elsewhere in the corpus
- entity swap: swap capitalized proper-noun-like tokens (people, places,
  organisations) for different entities drawn from elsewhere in the corpus
- sentence substitution: replace one interior sentence with an unrelated
  sentence from a different random article (a standard negative-sampling
  technique for fake-news datasets - mismatched claim in a genuine-sounding
  context)

Each real article gets exactly one synthetic fake counterpart, kept in the
SAME split as its source article (train/test/eval) to avoid leakage between
a real article and its own corrupted twin ending up in different splits.

This is a proxy, not organic misinformation, and the paper/report must state
that plainly - see notes/tshivenda-classifier-proxy.md.

Usage:
    python src/build_misinfo_proxy.py
    python src/build_misinfo_proxy.py --seed 7 --output /tmp/proxy.csv
"""

import argparse
import random
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = REPO_ROOT / "dataset" / "vukuzenzele" / "vukuzenzele-monolingual_ven.csv"
OUTPUT_CSV = REPO_ROOT / "dataset" / "vukuzenzele" / "misinfo_proxy_ven.csv"

NUMBER_RE = re.compile(r"\d+[.,]?\d*%?")
# capitalized tokens of 3+ letters, not at a sentence start (avoids picking up
# every sentence-initial word) - a simple, defensible proper-noun heuristic
ENTITY_RE = re.compile(r"(?<=[a-z,] )[A-Z][a-zA-Z]{2,}\b")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def load_clean(path):
    df = pd.read_csv(path)
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    return df


def build_pools(texts):
    numbers, entities = set(), set()
    for t in texts:
        numbers.update(NUMBER_RE.findall(t))
        entities.update(ENTITY_RE.findall(t))
    return sorted(numbers), sorted(entities)


def distort_numbers(text, number_pool, rng, rate=0.7):
    def repl(m):
        if rng.random() < rate and number_pool:
            return rng.choice(number_pool)
        return m.group(0)
    return NUMBER_RE.sub(repl, text)


def distort_entities(text, entity_pool, rng, rate=0.5):
    def repl(m):
        if rng.random() < rate and entity_pool:
            return rng.choice(entity_pool)
        return m.group(0)
    return ENTITY_RE.sub(repl, text)


def substitute_sentence(text, other_texts, rng):
    sentences = SENTENCE_SPLIT_RE.split(text)
    if len(sentences) < 3:
        return text  # too short to safely substitute an interior sentence
    donor = rng.choice(other_texts)
    donor_sentences = [s for s in SENTENCE_SPLIT_RE.split(donor) if len(s.split()) >= 4]
    if not donor_sentences:
        return text
    idx = rng.randrange(1, len(sentences))  # never replace the opening sentence
    sentences[idx] = rng.choice(donor_sentences)
    return " ".join(sentences)


def make_fake(text, all_texts, number_pool, entity_pool, rng):
    others = [t for t in all_texts if t != text]
    out = distort_numbers(text, number_pool, rng)
    out = distort_entities(out, entity_pool, rng)
    out = substitute_sentence(out, others, rng)
    return out


def build(input_csv, output_csv, seed):
    rng = random.Random(seed)
    df = load_clean(input_csv)
    all_texts = df["text"].tolist()
    number_pool, entity_pool = build_pools(all_texts)
    print(f"clean articles: {len(df)} | number pool: {len(number_pool)} | "
          f"entity pool: {len(entity_pool)}")

    rows = []
    for i, row in df.iterrows():
        rows.append({"id": f"real_{i}", "title": row["title"], "text": row["text"],
                     "label": 1, "split": row["split"], "source_id": i})
        fake_text = make_fake(row["text"], all_texts, number_pool, entity_pool, rng)
        rows.append({"id": f"fake_{i}", "title": row["title"], "text": fake_text,
                     "label": 0, "split": row["split"], "source_id": i})

    out = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"wrote {len(out)} rows ({len(df)} real + {len(df)} synthetic fake) -> {output_csv}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(INPUT_CSV))
    parser.add_argument("--output", default=str(OUTPUT_CSV))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.input, args.output, args.seed)
