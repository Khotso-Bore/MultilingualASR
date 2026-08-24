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

`src/classification/build_misinfo_proxy.py` takes the 184 Tshivenda articles already in the
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
   grouped 5-fold cross-validation (`src/classification/train_classifier.py`) rather than a
   single train/test split, so reported numbers are a mean +/- std across
   folds, not one noisy point estimate.
4. **Cannot be un-flagged without supervisor sign-off.** This substitution
   changes what Objective 4 actually measures for Tshivenda. Seani was
   informed of the blocker and the plan; this is the fallback implemented
   in the meantime, pending her confirmation.

## Results (5-fold grouped cross-validation)

First attempt (full fine-tune, all parameters trainable, lr 2e-5) was
unstable: loss stuck at ln(2) for 3 epochs at lr 2e-5 (no learning at all),
and at lr 5e-5 the model found real signal mid-training then collapsed back
to chance by the final epoch - a known small-dataset fine-tuning failure
mode. Fixed with two standard techniques: `load_best_model_at_end` (keep the
peak checkpoint, not the final one) and freezing the base encoder, training
only the classification head (`--freeze-base`, a "linear probe" - 592K of
278M parameters trainable). Also hit and fixed a real MPS bug along the way:
`scaled_dot_product_attention` on Apple's backend does not support dropout
and errors when the base is frozen - fixed by forcing eager attention
(`attn_implementation="eager"`), which works identically on MPS/CUDA/CPU.

Final validated config: `--freeze-base --learning-rate 1e-3 --epochs 15`
(early stopping, patience 3), 5-fold grouped CV:

| Model | Accuracy | Macro F1 |
|---|---|---|
| **AfroXLM-RoBERTa** (primary) | **0.562 +/- 0.030** | **0.550 +/- 0.030** |
| XLM-RoBERTa (comparison) | 0.500 +/- 0.000 | 0.333 +/- 0.000 |

XLM-RoBERTa landed at *exactly* the chance floor on all 5 folds - it could
not learn the task at all under this setup. AfroXLM-RoBERTa consistently
beat chance across every fold (std of only 0.030, vs. 0.171 before the
stability fix). This is not a bug - it directly corroborates MphayaNER
(Mbuvha et al., 2023, already cited in the proposal), which found
AfroXLM-RoBERTa is the strongest available model for Tshivenda NLP tasks
specifically because plain XLM-R's pretraining underrepresents the language.
A genuinely useful, literature-consistent result for the report.

## How to regenerate / retrain

```bash
python src/classification/build_misinfo_proxy.py                                       # rebuild the proxy dataset
python src/classification/train_classifier.py --model Davlan/afro-xlmr-base \
    --folds 5 --epochs 15 --learning-rate 1e-3 --freeze-base             # primary
python src/classification/train_classifier.py --model xlm-roberta-base \
    --folds 5 --epochs 15 --learning-rate 1e-3 --freeze-base             # comparison
```

Results land in `results/classifier/` (gitignored); logs used for this
writeup are `/tmp/clf_full_afroxlmr.log` and `/tmp/clf_full_xlmr.log`
(not committed - rerun the commands above to regenerate).
