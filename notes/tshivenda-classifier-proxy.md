# Tshivenda Misinformation Classifier: Proxy Dataset (Mukwevho Substitute)

## Why this exists

The proposal's Tshivenda sub-study (§4.5) specifies fine-tuning the
misinformation classifier on the Mukwevho et al. (2024) dataset - 23
labeled Tshivenda news articles, referenced throughout the proposal as the
GRU-baseline comparison point. That dataset is confirmed unavailable to this
team: it was never publicly released (the paper itself states the source is
"undisclosed"), it is not on the DSFSI public dataset registry, Hugging Face,
or GitHub, and the authors could not provide it on request.

The proposal already anticipates exactly this situation for the other two
languages - §3.2: *"For Setswana and Sepedi, no dedicated misinformation
datasets currently exist, and this project uses news topic classification
datasets as downstream proxy tasks."* Tshivenda was meant to be the
exception with a real misinformation dataset; without it, Tshivenda falls
back to the same proxy-task strategy, adapted to what is actually available.

No public Tshivenda topic-classification or misinformation dataset exists
either (checked: Hugging Face Hub broadly, MasakhaNEWS - 16 languages, no
Venda - and AfriSenti - 15 languages, no Venda). So the proxy had to be
constructed rather than found.

## What was built

`src/build_misinfo_proxy.py` takes the 184 Tshivenda articles already in the
repo (`dataset/vukuzenzele/`, the DSFSI Vukuzenzele government-magazine
corpus, CC-BY-4.0) and constructs a balanced real/fake classification task:

- **label = 1 (real)**: the original article, verbatim.
- **label = 0 (fake)**: a synthetic counterpart built by distorting the same
  article's numbers (statistics, percentages, dates), swapping capitalized
  entities (people, places, organisations) for different entities drawn
  from elsewhere in the corpus, and substituting one interior sentence with
  an unrelated sentence from a different random article. This mirrors how
  real misinformation typically works: a genuine-sounding article with
  wrong facts, not gibberish.

Result: 358 rows (179 real + 179 synthetic fake), saved to
`dataset/vukuzenzele/misinfo_proxy_ven.csv`.

## What was verified before use

- Balanced classes (179/179).
- **Zero split leakage**: a real article and its own synthetic fake always
  stay in the same train/test/eval split (grouped by source article id) -
  otherwise the classifier could trivially "cheat" by recognising a
  near-duplicate it saw during training.
- **No length confound**: real vs. fake word-count distributions are nearly
  identical (mean 749 vs. 749 words) - the classifier cannot win by learning
  "fake articles are shorter/longer," it has to actually read content.
- Tshivenda diacritics (ḓ ḽ ṅ ṋ ṱ) preserved in all 358 rows.
- Deterministic given a fixed seed.

## Honest limitations (state these plainly in the report)

1. **This is synthetic, not organic, misinformation.** Real Tshivenda
   misinformation (rumour, political disinformation, health myths) has
   different linguistic patterns than fact-swapped government prose. The
   classifier is learning to detect *this specific corruption method*, not
   misinformation in general - a real limitation of the proxy, same as the
   news-topic-classification proxies already used for Setswana/Sepedi are
   not really "misinformation" tasks either.
2. **No comparison to the Mukwevho GRU baseline is possible** - that
   requires the original dataset. Objective 4's "compared to existing
   text-only baselines" is satisfied by comparing AfroXLM-RoBERTa against
   XLM-RoBERTa on this proxy instead, not against Mukwevho's GRU.
3. **Small source pool**: only 179 unique articles, all formal government
   register (Vukuzenzele), none of the informal/social-media register real
   misinformation typically appears in. Addressed as best possible via
   grouped 5-fold cross-validation (`src/train_classifier.py`) rather than a
   single train/test split, so reported numbers are a mean +/- std across
   folds, not one noisy point estimate.
4. **Cannot be un-flagged without supervisor sign-off.** This substitution
   changes what Objective 4 actually measures for Tshivenda. Seani was
   informed of the blocker and the plan; this is the fallback implemented
   in the meantime, pending her confirmation.

## How to regenerate / retrain

```bash
python src/build_misinfo_proxy.py                          # rebuild the proxy dataset
python src/train_classifier.py --model Davlan/afro-xlmr-base   # primary
python src/train_classifier.py --model xlm-roberta-base        # comparison
```

Results land in `results/classifier/` (gitignored); see
`notes/pilot-ven-results.md` for the numbers once training completes.
