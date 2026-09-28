# Tshivenda Demo Pipeline: Audio In, Verdict Out

Answers a question that came up directly: does this project actually have a
speech-recognition-for-misinformation-detection pipeline, or two separate
things that only get compared on paper? Until this, it was the second one.
This is the first place real audio actually flows through both models to
produce one verdict, and the first time either model has been run at volume
outside a fixed, curated evaluation set.

## Why this needed building

Everywhere else in the repo, the ASR model and the misinformation classifier
are connected only through a simulation:
`notes/tshivenda-error-propagation.md` corrupts clean article text by a
measured error rate and checks how the classifier reacts to that. No real
audio was ever actually transcribed and then handed to the classifier.

## What was built

Two new scripts, `src/demo/`:

- `asr_to_classifier_pipeline_ven.py` - one clip in, one verdict out. Loads
  the fine-tuned Whisper checkpoint (`results/whisper-ven-pilot-v2/final`,
  the full-scale 0.103 WER model - see `notes/pilot-ven-results.md`),
  transcribes, then loads the misinformation classifier and classifies the
  transcript. Can pick a random real clip from the NCHLT test / ANV
  dev_test sets, or take any `.wav` path.
- `run_pipeline_batch_ven.py` - the same two models loaded once, run over
  many real clips in a loop, one CSV row per clip (corpus, audio path,
  reference, ASR hypothesis, WER, CER, classifier label, confidence).
  Flushes to disk after every row, and supports `--resume` so an
  interruption doesn't lose work already done.

One new saved artifact this needed: `results/classifier/final`, an actual
deployable classifier checkpoint. `train_classifier_ven.py`'s 5-fold
cross-validation never kept one - each fold reused the same temp output
directory and got overwritten by the next fold, which was fine for an
accuracy estimate but left nothing to load for inference. Added
`--save-final DIR` to that script: trains once on a 90/10 grouped split
(same proven config, frozen base, `lr 1e-3`, 15 epochs, early stopping) and
saves the result. Held-out check on that 15% split: accuracy 0.648, macro F1
0.642 (n=54) - in the same range as the 5-fold CV numbers, not a broken
checkpoint.

## The honest limit on what this proves

The classifier was trained on long Vukuzenzele government articles - real
ones plus fact-distorted synthetic fakes (`notes/tshivenda-classifier-proxy.md`).
The NCHLT and ANV audio clips are short single-sentence recordings from a
speech corpus, not recordings of those articles. No audio of the actual
real/fake proxy content exists, and there is no Tshivenda text-to-speech
model to make any (checked: no `facebook/mms-tts-ven`, nothing else on
HuggingFace covers Venda TTS).

So a verdict on a transcribed NCHLT/ANV clip is a real label with a real
confidence score, and it proves the two models mechanically chain together,
but it is not a scored "correct/incorrect" prediction - there is no ground
truth for whether a random read sentence is "real" or "fake"
misinformation. The only accuracy number that means anything for Objective 4
is still 0.562, from the text-only proxy dataset. What follows is what a
batch run at real volume actually shows.

## The run

1,000 real clips (500 from NCHLT test, 500 from ANV dev_test, random sample,
seed 42), each one transcribed and classified individually through a live
pipeline, not the batched evaluation used elsewhere in this repo. Took about
45 minutes of active generation time once model loading was excluded (the
per-clip rate visibly dropped partway through - see "A real robustness
finding" below for why).

### ASR side: real WER at 500-clip volume

| Corpus | n | Mean WER | Mean CER |
|---|---|---|---|
| NCHLT test | 500 | 0.114 | 0.032 |
| ANV dev_test | 500 | 0.348 | 0.178 |

Close to, but not identical to, the standardized 200-clip fixed-seed numbers
reported elsewhere (0.103 NCHLT / 0.256 ANV) - expected, since this is a
different, larger, freshly-sampled set of clips, not the same fixed
comparison set the rest of the report uses. The NCHLT gap is small (0.114 vs
0.103). The ANV gap is bigger (0.348 vs 0.256), and it has a specific,
findable cause below rather than just being noise.

### A real robustness finding: runaway generation on longer clips

8 of the 500 ANV clips (1.6%) came back with WER above 1.0 - the model
generated far more words than the reference contains, a repetition-loop
failure mode Whisper is known for on longer or harder audio. One example:
reference is 23 words, the generated hypothesis is 222 words, WER 9.17.
Another: reference 70 words, hypothesis 210 words, WER 2.39, ending in the
same phrase repeated over and over:

`...iyo iyo thithi nṱhe yawe ya tsini hawe ya tsini hawe ya tsini hawe ya
tsini hawe ya tsini hawe ya tsini hawe ya tsini hawe ya tsini hawe ya tsini
hawe ya tsini hawe ya tsini hawe ya tsini hawe...` (repeats dozens more
times)

These 8 outlier clips, 1.6% of the ANV sample, are why the ANV mean WER
looks worse here than the standardized eval: excluding them, ANV mean WER
drops to 0.193, much closer to the 0.256 standardized number. This also
explains something visible in the raw run log
(`results/logs/demo_pipeline_batch_1000clips.log`) - the processing rate
dropped from about 0.36 clips/second to about 0.12 clips/second partway
through the run, exactly when NCHLT clips (shorter, first 500) gave way to
ANV clips (longer, spontaneous speech, last 500) - a repetition-loop clip
takes far longer to generate because the model keeps producing tokens until
it hits the max-length cap, not because anything is wrong with the machine.

This is a genuine finding the fixed 200-clip standardized eval sets used
everywhere else in this report would very plausibly miss or under-sample,
purely because they are 200 clips, not 500 - it took running at real volume
to surface a real, reproducible failure mode worth naming honestly rather
than averaging away.

### What the transcripts actually look like

**Clean** (WER 0.0, NCHLT):
```
reference:  humiselwa kha muiti wa khumbelo
hypothesis: humiselwa kha muiti wa khumbelo
```

**Typical/mid-range** (WER 0.128, ANV):
```
reference:  ndi nga ka maḓi a manzhi nda dzula ndo a vhea nda dovha hafhu
            nda lingedza u gwa maḓi fhasi hune nda kho vhona uri a kho bva
            hone nda u tsireledza u itela uri ndi kone u fha zwifuwo zwanga
            ndi kone u sheledza na mitshelo
hypothesis: ndi nga kamaḓi a manzhi nda dzula ndo a vhea nda dovha hafhu
            nda lingedza u gwa maḓifhasi hune nda khou vhona uri a khou bva
            hone nda u tsireledza u itela uri ndi kone u fha zwifuwo zwanga
            ndi kone u sheledza na mitshelo
```
Word-boundary merges (`ka maḓi` -> `kamaḓi`, `maḓi fhasi` -> `maḓifhasi`)
and `kho`/`khou` slips - minor, still fully readable, same pattern already
documented in `notes/pilot-ven-results.md`.

**Runaway generation** (WER 9.17, ANV, the worst clip in the run):
```
reference (23 words):  [short utterance]
hypothesis (222 words): [same phrase repeated dozens of times]
```
See the repetition example above for the full text.

### Classifier side: verdict distribution and confidence

| | n | real | fake | % fake |
|---|---|---|---|---|
| NCHLT test | 500 | 229 | 271 | 54.2% |
| ANV dev_test | 500 | 463 | 37 | 7.4% |
| **Overall** | **1000** | **692** | **308** | **30.8%** |

Mean confidence 0.658 overall (real: 0.670, fake: 0.632) - the classifier is
never close to a coin flip on these, but see the caveat below on why that
alone doesn't mean much here.

**A sharp corpus split worth flagging, not a misinformation finding**: NCHLT
transcripts get called "fake" 54.2% of the time, ANV transcripts only 7.4%
of the time. Mean WER also splits the same way by verdict (fake-verdict
clips: mean WER 0.115, real-verdict clips: mean WER 0.283) - but WER and
verdict are both downstream of the same underlying cause, not causally
linked to each other. NCHLT clips are short, clipped, single-sentence reads.
ANV clips are longer, spontaneous, conversational speech, which reads
stylistically closer to the long-form Vukuzenzele prose the classifier was
actually trained on. The likely explanation is the classifier picking up on
sentence length/register, not anything about truthfulness - exactly the
domain-mismatch limitation already stated above, now visible as a concrete
pattern in real output rather than just an abstract caveat.

## Caveats

- No ground truth for these verdicts (see "The honest limit" above) - this
  section reports a distribution and a robustness finding, not an accuracy
  number.
- The runaway-generation failure mode was found in 1.6% of one 500-clip ANV
  sample - real and reproducible (the log timing confirms it, not just the
  WER numbers), but this is a single run at one seed, not a swept estimate
  of exactly how often it happens.
- `results/classifier/final`'s single 90/10 held-out check (n=54) is a
  sanity check that the saved checkpoint works, not a replacement for the
  5-fold CV estimate already reported in `notes/tshivenda-classifier-proxy.md`.
- The full per-clip CSV (`results/demo_pipeline/pipeline_batch_results.csv`)
  is not committed, same as every other per-clip prediction CSV in this repo
  (`results/*` is gitignored except `results/logs/`) - the raw run log is
  the tracked evidence trail instead.

## How to reproduce

```bash
# save a deployable classifier checkpoint (one-time, ~10 min)
python src/classification/train_classifier_ven.py --model Davlan/afro-xlmr-base \
    --freeze-base --learning-rate 1e-3 --epochs 15 --save-final results/classifier/final

# one real clip
python src/demo/asr_to_classifier_pipeline_ven.py --random-clip --seed 42

# batch, real volume (~45 min of generation time for 1000 clips on M4 MPS)
python src/demo/run_pipeline_batch_ven.py --limit 500 --seed 42 \
    --out results/demo_pipeline/pipeline_batch_results.csv
```

Logs used for this writeup: `results/logs/classifier_final_checkpoint_train.log`,
`results/logs/demo_pipeline_batch_1000clips.log`.
