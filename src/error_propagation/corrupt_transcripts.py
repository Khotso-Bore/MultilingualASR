"""Controlled-WER transcript corruption for the error-propagation study (§4.6).

Takes clean Tshivenda transcripts and produces corrupted versions at a target
Word Error Rate, using word-level operations that mirror how ASR actually fails
on Tshivenda (patterns observed in the zero-shot Whisper baseline):

- substitution: prefer phonetically-plausible perturbations of the original word
  (diacritic dropping ṱ->t / ḓ->d / ḽ->l / ṅ,ṋ->n, cluster shifts tsh->ch,
  vh->w, fh->p, dzh->j, vowel swaps), falling back to corpus-vocabulary swaps.
  If an error model measured from real ASR output is supplied, real confusion
  pairs take priority.
- deletion: drop a word.
- insertion: insert a word sampled from the corpus unigram distribution.

All output stays within the 32-character Tshivenda set (dropping a diacritic is
allowed - that is exactly what real ASR output does - but nothing outside the
set is ever introduced).

Default mixed-mode S:D:I ratio is 60:25:15 (substitution-heavy, typical of a
usable ASR system). Measure the real ratio from fine-tuned model output later
via zero_shot_baseline.py --save-predictions and pass it with --error-model.

Usage:
    python src/error_propagation/corrupt_transcripts.py --input dataset/processed/nchlt_ven/test.csv \
        --target-wer 0.3 --mode mixed --seed 42 --output /tmp/corrupted_30.csv
    python src/error_propagation/corrupt_transcripts.py ... --mode substitution   # ablation
    python src/error_propagation/corrupt_transcripts.py ... --error-model results/zero_shot/large-v3_nchlt_test.csv
"""

import argparse
import csv
import random
import re
from collections import Counter
from pathlib import Path

ALLOWED = set(" abcdefghijklmnopqrstuvwxyzḓḽṅṋṱ")

# Phonetically-plausible perturbation rules, mined from the zero-shot baseline's
# failure modes (applied left to right, first applicable subset is used)
PHONETIC_RULES = [
    ("ṱ", "t"), ("ḓ", "d"), ("ḽ", "l"), ("ṅ", "n"), ("ṋ", "n"),
    ("tsh", "ch"), ("vh", "w"), ("fh", "p"), ("dzh", "j"), ("zw", "z"),
    ("kh", "k"), ("th", "t"), ("ph", "p"),
]
VOWELS = "aeiou"

DEFAULT_RATIO = {"substitution": 0.60, "deletion": 0.25, "insertion": 0.15}


def _stochastic_round(x, rng):
    base = int(x)
    return base + (1 if rng.random() < (x - base) else 0)


def phonetic_perturb(word, rng):
    """Return a perturbed variant of word, or None if no rule applies."""
    applicable = [(a, b) for a, b in PHONETIC_RULES if a in word]
    rng.shuffle(applicable)
    for a, b in applicable:
        out = word.replace(a, b, 1)
        if out != word and out:
            return out
    # vowel swap fallback
    positions = [i for i, c in enumerate(word) if c in VOWELS]
    if positions:
        i = rng.choice(positions)
        repl = rng.choice([v for v in VOWELS if v != word[i]])
        out = word[:i] + repl + word[i + 1:]
        if out != word:
            return out
    return None


class ErrorModel:
    """Optional statistics measured from real ref/hyp ASR output pairs."""

    def __init__(self, ratio=None, confusions=None):
        self.ratio = ratio or dict(DEFAULT_RATIO)
        self.confusions = confusions or {}  # ref_word -> [hyp_word, ...]

    @classmethod
    def from_prediction_files(cls, paths):
        import jiwer
        refs, hyps = [], []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["reference"].strip():
                        refs.append(row["reference"])
                        hyps.append(row["hypothesis"])
        out = jiwer.process_words(refs, hyps)
        total = out.substitutions + out.deletions + out.insertions
        ratio = dict(DEFAULT_RATIO)
        if total:
            ratio = {
                "substitution": out.substitutions / total,
                "deletion": out.deletions / total,
                "insertion": out.insertions / total,
            }
        confusions = {}
        for chunks, ref_sent, hyp_sent in zip(out.alignments, refs, hyps):
            rw, hw = ref_sent.split(), hyp_sent.split()
            for c in chunks:
                if c.type == "substitute":
                    for r, h in zip(rw[c.ref_start_idx:c.ref_end_idx],
                                    hw[c.hyp_start_idx:c.hyp_end_idx]):
                        if set(h) <= ALLOWED and h != r:
                            confusions.setdefault(r, []).append(h)
        return cls(ratio=ratio, confusions=confusions)


def corrupt_sentence(sentence, target_wer, mode, rng, unigrams, error_model):
    words = sentence.split()
    n = len(words)
    if n == 0:
        return sentence

    n_errors = _stochastic_round(target_wer * n, rng)
    if n_errors == 0:
        return sentence

    if mode == "mixed":
        ratio = error_model.ratio
        n_sub = _stochastic_round(n_errors * ratio["substitution"], rng)
        n_del = _stochastic_round(n_errors * ratio["deletion"], rng)
        n_ins = n_errors - n_sub - n_del
        if n_ins < 0:
            n_sub = max(0, n_sub + n_ins)
            n_ins = 0
    else:
        n_sub = n_errors if mode == "substitution" else 0
        n_del = n_errors if mode == "deletion" else 0
        n_ins = n_errors if mode == "insertion" else 0

    # substitutions/deletions cannot exceed available words; overflow -> insertions
    if n_sub + n_del > n:
        overflow = n_sub + n_del - n
        take_from_del = min(overflow, n_del)
        n_del -= take_from_del
        overflow -= take_from_del
        n_sub -= overflow
        n_ins += 0 if mode != "mixed" else 0  # ablations stay pure; mixed just caps

    positions = list(range(n))
    rng.shuffle(positions)
    sub_pos = set(positions[:n_sub])
    del_pos = set(positions[n_sub:n_sub + n_del])

    out = []
    for i, w in enumerate(words):
        if i in del_pos:
            continue
        if i in sub_pos:
            new = None
            cands = error_model.confusions.get(w)
            if cands:
                new = rng.choice(cands)
            if new is None or new == w:
                new = phonetic_perturb(w, rng)
            if new is None or new == w:
                pool = [u for u in unigrams if u != w]
                new = rng.choice(pool) if pool else w
            out.append(new)
        else:
            out.append(w)

    for _ in range(n_ins):
        out.insert(rng.randrange(len(out) + 1), rng.choice(unigrams))

    return " ".join(out)


def corrupt_file(input_csv, output_csv, target_wer, mode, seed, error_model=None,
                 text_column="transcript"):
    rng = random.Random(seed)
    error_model = error_model or ErrorModel()

    rows = list(csv.DictReader(open(input_csv, encoding="utf-8")))
    counts = Counter()
    for row in rows:
        counts.update(row[text_column].split())
    unigrams = [w for w, _ in counts.most_common(2000)]

    out_rows = []
    for row in rows:
        row = dict(row)
        row["transcript_clean"] = row[text_column]
        row[text_column] = corrupt_sentence(row[text_column], target_wer, mode,
                                            rng, unigrams, error_model)
        out_rows.append(row)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    return out_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with a transcript column")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-wer", type=float, required=True)
    parser.add_argument("--mode", default="mixed",
                        choices=["mixed", "substitution", "deletion", "insertion"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--error-model", nargs="*", default=None,
                        help="ref/hyp prediction CSVs from zero_shot_baseline.py --save-predictions")
    parser.add_argument("--text-column", default="transcript")
    args = parser.parse_args()

    em = ErrorModel.from_prediction_files(args.error_model) if args.error_model else None
    corrupt_file(args.input, args.output, args.target_wer, args.mode, args.seed,
                 error_model=em, text_column=args.text_column)
    print(f"corrupted {args.input} at target WER {args.target_wer} ({args.mode}) -> {args.output}")
