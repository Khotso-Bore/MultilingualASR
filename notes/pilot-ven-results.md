# Tshivenda MPS Pilot Fine-Tune - Results

Pilot: Wav2Vec2 XLS-R-300M, 5,000 NCHLT train clips (<= 10 s), 3 epochs, on an
M4 MacBook (MPS). NOT the real Stage 1 (60k clips, 10 epochs, GPU) - a proof
that fine-tuning moves the needle, run while GPU access is pending.
Training: 100 min, loss 38 -> 8.7, no divergence.

## WER/CER vs zero-shot (200 clips per corpus, seed 42, same normalisation)

| Model | NCHLT test WER | NCHLT CER | ANV dev_test WER | ANV CER |
|---|---|---|---|---|
| Whisper Large v3 zero-shot | 1.108 | 0.763 | 1.072 | 0.501 |
| Pilot Wav2Vec2 (5k clips, 3 ep) | **0.614** | **0.151** | **0.851** | **0.257** |

- NCHLT CER 15%: the model is clearly learning Tshivenda orthography
  (e.g. "ya u sumbedzwa tshirunzi na" -> "ya u sumbedzwa tshirundzi na").
- ANV lags (out-of-domain for this pilot: trained on NCHLT read speech only,
  ANV is scripted/unscripted spontaneous speech with longer clips).

## Diacritics finding (honest correction)

The pilot emits ZERO diacritics (ḓ ḽ ṅ ṋ ṱ): 757 occurrences in references,
0 in predictions - it writes the plain-letter equivalent (nṋe -> ne,
ṱuṱuwedzwa -> totowedzwa). The vocabulary supports them; 5k clips / 3 epochs
is not enough for the model to learn these rare characters. Whether they
emerge is a key thing to check after the real Stage 1 run.

## Measured error model (first real fine-tuned-ASR error statistics for Tshivenda)

From 400 saved ref/hyp pairs (results/preds_pilot/, gitignored):

- S:D:I ratio = **49.5 : 49.3 : 1.3** (vs the corruption engine's assumed
  60:25:15 default) - real errors are far more deletion-heavy, insertions
  nearly absent. Use via corrupt_transcripts.py --error-model.
- Top confusion pairs are linguistically plausible: nṋe->ne, zwine->zine,
  hu<->u, wana->wa (diacritic loss and cluster simplification dominate).

## Caveats

- Pilot only: small subset, short-clip cap, single corpus, few epochs.
- Deletion-heavy ratio partly reflects degenerate merged-word output on long
  ANV clips; re-measure after Stage 1.
- Regenerate predictions: src/evaluate_wav2vec2.py --checkpoint
  results/wav2vec2-ven-pilot/final --save-predictions results/preds_pilot
