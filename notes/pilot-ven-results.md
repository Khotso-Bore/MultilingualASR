# Tshivenda MPS Pilot Fine-Tune - Results

## Whisper full-scale (final) - best ASR result overall, answers Objective 1

`src/asr/pilot_finetune_whisper_mps_ven.py --resume-from results/whisper-ven-pilot-v2/final
--include-anv --epochs 1 --learning-rate 5e-5 --train-clips 100000 --eval-clips 500`,
resumed from pilot v2 (0.182/0.048), full NCHLT+ANV train pool (60,087 raw
clips), 1 epoch, M4 MacBook (MPS), 11h21m34s training (~12.5h total with
preprocessing - see `notes/whisper-full-scale-run-log.md` for the live run
log). This is the real Objective 1 / Sub-question 1 headline number - full
scale, not another laptop pilot. Checkpoint overwrote
`results/whisper-ven-pilot-v2/final` (the resume-naming logic reuses the
same `-v2` suffix regardless of how many times it's resumed from itself, so
that path now holds this run's weights, not the smaller pilot's).

Training-time eval (mixed NCHLT-validation + ANV-dev, not the fixed 200-clip
test sets below): WER 0.111 / CER 0.032.

Comparison (200-clip NCHLT test / ANV dev_test, seed 42 - same fixed sets as
every other row in this file):

| Model | NCHLT WER | NCHLT CER | ANV WER | ANV CER |
|---|---|---|---|---|
| Whisper Large v3 zero-shot | 1.108 | 0.763 | 1.072 | 0.501 |
| Wav2Vec2 pilot v2 (7,325 clips, 8 total epochs) | 0.332 | 0.074 | 0.537 | 0.127 |
| Whisper pilot v1 (5k NCHLT, 3 epochs) | 0.265 | 0.060 | - | - |
| Whisper pilot v2 (7,325 clips, resumed) | 0.182 | 0.048 | - | - |
| **Whisper full-scale (60k pool, resumed, 1 epoch)** | **0.103** | **0.032** | **0.256** | **0.108** |

Roughly halves pilot v2's already-strong NCHLT WER (0.182 -> 0.103) and cuts
the ANV gap by more than half (0.537 -> 0.256) with one additional epoch on
~5x the data. Wav2Vec2 stays at its pilot-scale number as the comparison
point (see `notes/whisper-full-scale-run-log.md` for why only Whisper got
pushed to full scale - compute constraints, and Whisper was already the
clearly stronger architecture).

### Reference vs. hypothesis examples (full-scale checkpoint, real predictions)

Full data: `results/preds_full/final_nchlt_test.csv` / `final_anv_dev_test.csv`
(200 rows each, gitignored - regenerate with `zero_shot_baseline_ven.py
--model results/whisper-ven-pilot-v2/final --save-predictions results/preds_full`).
Rows below are in sampled order (seed 42), **not cherry-picked** - the good,
the mediocre, and the bad are all included so this is an honest picture, not
a highlight reel.

**Exact-match rate**: NCHLT 131/200 (65.5%), ANV 25/200 (12.5%) - the domain
gap between short read-speech and long spontaneous speech is real and large,
not just a WER-number abstraction.

**NCHLT test (WER 0.103)** - first 30 of 200:

| # | Reference | Hypothesis | Row WER |
|---|---|---|---|
| 1 | i fanela u dzhiela nzhele | i fanela u dzhiela nzhele *(exact)* | 0.00 |
| 2 | na vhuḓifhinduleli kha vhashumi nahone | na vhuḓifhinduleli kha vhashumi nahone *(exact)* | 0.00 |
| 3 | na u vhambedzea na dza | na u vhambedzea na dza *(exact)* | 0.00 |
| 4 | ya u sumbedzwa tshirunzi na | ya u sumbedzwa tshirunzi na *(exact)* | 0.00 |
| 5 | vhulimi zwine zwa khou bvelela | vhulimi zwine zwa khou bvelela *(exact)* | 0.00 |
| 6 | havhudi vhune ha sa tou | havhuḓi vhune ha sa tou | 0.20 |
| 7 | tsha kale musi vhasidzana vha | tshakale musi vhasidzana vha | 0.40 |
| 8 | humiselwa kha muiti wa khumbelo | humiselwa kha muiti wa khumbelo *(exact)* | 0.00 |
| 9 | oweleaho wa matombo a linton | owelaho wa matombo a ḽinthoni | 0.40 |
| 10 | zwa wela fhasi hadzo kha | zwa wela fhasi hadzo kha *(exact)* | 0.00 |
| 11 | wa tshelede ya u unḓa | wa tshelede ya u | 0.20 |
| 12 | na mugudisi wa u bambela | na mugudisi wa u bammbela | 0.20 |
| 13 | lwone holu lwanga lu a | lwone holu lwanga lu a *(exact)* | 0.00 |
| 14 | kona u ṅwala na u | kona u ṅwala na u *(exact)* | 0.00 |
| 15 | tambudzwa ndi nga u sedzulusa | tambudzwa ndi nga u sedzulusa *(exact)* | 0.00 |
| 16 | na vhuhole kana u thogomelwa | na vhuhole kana vhuṱhogomelwaho | 0.40 |
| 17 | u rekhoda kha redzhisitara ya | u rekhoda kha redzhisitara ya *(exact)* | 0.00 |
| 18 | nekedza tshumelo kha vhaaluwa ho | nekedza tshumelo kha vhaaluwa ho *(exact)* | 0.00 |
| 19 | a nga dzhia tsheo ya | a nga dzhia tsheo ya *(exact)* | 0.00 |
| 20 | lushaka hune ha vhonala na | lushaka hune ha vhonala na *(exact)* | 0.00 |
| 21 | vha ṋekane nga khophi yo | vha ṋekane nga khophi yo *(exact)* | 0.00 |
| 22 | kana a sa tsha takalela | kana a sa tsha takalela *(exact)* | 0.00 |
| 23 | elimi | ilimi | 1.00 |
| 24 | kana u khethulula zwi tshi | kana u khethulula zwi tshi *(exact)* | 0.00 |
| 25 | vhadzulapo vha tea u kwamiwa | vhadzulapo vha tea u kwamiwa *(exact)* | 0.00 |
| 26 | mirado ya tshigwada tsha nnda | mirado ya tshigwada tsha nnḓa | 0.20 |
| 27 | thoma wa kwamana na vhadzulapo | thoma wa kwamana na vhadzulapo *(exact)* | 0.00 |
| 28 | u takadza dzikhasitama namusi bannga | u takadza dzikhasitama namusi bannga *(exact)* | 0.00 |
| 29 | zwa dzinnu dza lushaka zwo | zwa dzinnu dza lushaka zwo *(exact)* | 0.00 |
| 30 | a si na vhukwamani na | a si na vhukwamani na *(exact)* | 0.00 |

Note row 23 (`elimi` -> `ilimi`, WER 1.00): a single-word clip is an
all-or-nothing WER score even though only one letter differs - a reminder
that word-level WER can overstate severity on very short clips. Rows 6, 9,
16, 26 show the typical error shape at this scale: one diacritic/cluster
slip (havhudi->havhuḓi, nnda->nnḓa) or a word-boundary shift, not garbled
output.

**ANV dev_test (WER 0.256)** - first 25 of 200 (harder domain: longer,
spontaneous speech; excerpts truncated to ~90 chars for table width, full
text in the CSV):

| # | Reference | Hypothesis | Row WER |
|---|---|---|---|
| 1 | khombo ndi musi vhukonani ho no kalula ni wane khonani yaṋu iṅwe hanefho online a tshi... | khombo ndi musi vhukonani ho no kalula ni wane khonani yaṋu iṅwe hanefho online a tshi... | 0.18 |
| 2 | ahuna khaelo na nthihi ine ya ṋetshedza tsireledzo yo fhelelaho tsireledzo ya percent | ahuna khaelo na nthihi ine ya ṋetshedzwa tsireledzo yo fhelelaho tsireledzo ya percent | 0.08 |
| 3 | ee nṋe ndi soko vhona unga vhaswa vha hune nda dzula hone vha funa zwiambaro zwa musala... | ee nṋe ndi soko vhona unga vhaswa vha hune nda dzula hone vha funa zwiambaro zwa musala... | 0.18 |
| 4 | u sa tsireledzea ha zwiḽiwa zwi tshi khou ṱuṱuwedzwa nga kiḽima zwi ita uri mutakalo wa... | u sa tsireledzea ha zwiḽiwa zwi tshi khou ṱuṱuwedzwa nga kilima zwi ita uri mutakalo wa... | 0.13 |
| 5 | nṋe ndi vhona unga maapuḽa dzi banana maswiri mapierre dzi nḓirivhe ndi magwavha ndi vh... | nṋe ndi vhona u nga maapula dzi banana maswiri maphiere dzi nḓirivhe ndi magwavha ndi v... | 0.24 |
| 6 | fhedziha zwenezwi miṅwaha i tshi khou ḓi ya o ḓo shandukisa muhumbulo musi a tshi vhona... | fhedziha zwenezwi miṅwaha i tshi khou ḓi ya o ḓo shandukisa muhumbulo musi a tshi vhona... *(exact)* | 0.00 |
| 7 | ro pembela nga maḓikiṱa midavhini hombo ḓivha na ḓikiṱa ḽihulu nga ḓuvha ḽi tevhelaho ḽ... | ro pembela nga maḓiki ṱa midavheni ho vha ho ṱangana vhathu vho fhambananaho na vharang... | **0.64** |
| 8 | kha vhupo ha hashu mbudzi dzi shumiswa kha tshisevho vhaṅwe vha shumisa na kha mafhi vh... | kha vhupo ha hashu ngudzo dzi shumiswa kha tshisevho vhaṅwe vha shumisana kha maanḓa vh... | 0.21 |
| 9 | ee ngeno mahayani zwo ḓowelea ngeno u tshi fanela u to buba uri u kone u vha kha vhuimo... | ee ngeno mahayani zwo ḓowelea ngeno u tshi fanela u tou buba uri u kone u vha kha vhuim... | 0.05 |
| 10 | ee vhuponi hashu dzi hone dzi kiḽiniki dza dzimobaiḽi dzine dza ḓa dzi ngavha dzi tshi... | ee vhuponi ha hashu dzi hone dzi kiḽiniki dza dzi mobaiḽi dzine dza ḓa dzi nga vha dzi... | 0.27 |
| 11 | i si gathi yo fhiraho vharengi vha afrika | i si gathi yo fhiraho vharengi vha afrika *(exact)* | 0.00 |
| 12 | ndi vhugai ine muswa a ḓo hola yone | vhugai ine muswa a ḓo hola yone | 0.12 |
| 13 | ndi u ḓisa thekhinoḽodzhi i leludzaho kha uri vhalwadze vha kone u wana faila dzavho ng... | ndi u ḓisa thekhinolodzhi iyo leludzaho kha uri vhalwadze vha kone u wana fhaela dzavho... | 0.17 |
| 14 | vhaṅwe vha vha vha kho itela u dzi ḓivha musi dzo no ṱangana na dziṅwe u itela uri vhas... | vhaṅwe vha vha khou itelwa u dzi ḓivha musi dzo no ṱangana na dziṅwe u itela uri vha si... | 0.23 |
| 15 | nṋe ndi vhona unga muthu u fanela u vha o ṱamba zwanḓa a tshetshelela ṋama fhethu ho ku... | nṋe ndi vhona u nga muthu u fanela u vha o ṱamba zwanḓa a tshi tshelela ṋama fhethu ho... | 0.17 |
| 16 | khaelo iṅwe na iṅwe yo shumiswaho kha mbekanyamushumo ya u haela vhathu vhanzhi afrika... | khaelo iṅwe na iṅwe yo shumiswaho kha mbekanyamushumo ya u haela vhathu vhanzhi afrika... | 0.10 |
| 17 | vho humbela vho ramabindu maṱuku u ita tshipiḓa tshavho sa izwi vha na tshikhala tsha u... | vho humbela vhoramabindu maṱuku u ita tshipiḓa tshavho sa izwi vha na tshikhala tsha u... | 0.08 |
| 18 | u ya wana vhenevho vhathu vhane vha kona u runga vha tshi ṱoḓana na dzi rokho dzenedzi... | u ya wana vhane vha vhathu vhane vha kona u hunga vha tshi ṱoḓa na na dzi rogo dzine dz... | **0.45** |
| 19 | tshanduko kha mitengo ya zwiḽiwa yo rekhodiwa u vha henefha kha phesenthe dza | tshanduko kha mitengo ya zwiḽiwa yo rekhodiwa u vha henefho kha phesenthe dza | 0.08 |
| 20 | nṋe ndi ḓivha best med fedhealth nda dovha nda ḓivha momentum discovery na ya muvhuso g... | nṋe ndi ḓivha based aid um ndi ḓivha ndi ḓivha miṱhamu ndi si khavha ri na ya mvhuso ge... | **0.63** |
| 21 | ndi vhona i songo fanela nga uri vhathu vha ḓo fhedzisela vha tshi vho ri itela zwithu... | ndi vhona i songo fanela ngauri vhathu vha ḓo fhedzisela vha tshi vho ri itela zwithu... | 0.12 |
| 22 | nga o ḓo kona u rengisa mashango a nnḓa thani dzi swikaho dza zwikavhavhe nahone nga ts... | nga o ḓo kona u rengisa mashango a nnḓa thani dzi swikaho dza zwikavhavhe nahone nga ts... | 0.14 |
| 23 | ndi nga luvhelela muthu uyo ane a kho nthusa uri a shumise ine ya vha kha lutingo lwang... | ndi nga luvhele dza muthu uyo ane a khou nthusa uri a shumise ine yavha kha luṱingo lwa... | 0.34 |
| 24 | vho ṱalutshedza uri muhasho u dzhia maitele o fhelelaho na u ṋetshedza nyengedzedzo yo... | vho talutshedza uri muhasho u dzhia maitele o fhelelaho na u ṋetshedza nyengedzedzo yo... | 0.24 |
| 25 | u ya nga vha muhasho wa pfunzo dza mutheo dbe vhagudi vha a funzwa na nga ha mikhwa yav... | u ya nga vha muhasho wa pfunzo dza mutheo dbe vhagudi vha a funzwa na nga ha mikhwa yav... *(exact)* | 0.00 |

The bad cases are real, not swept under the rug: **rows 7 and 20 (WER 0.64
and 0.63)** are genuine content divergence, not just word-boundary noise -
row 20 in particular ("best med fedhealth ... momentum discovery" ->
"based aid um ... miṱhamu") looks like the model struggling with
code-switched English brand names embedded in Tshivenda speech, which this
project's training data doesn't specifically target. Row 18 (0.45) mixes
real substitutions with a garbled clause. These are the honest tail of the
ANV distribution, not the norm (12.5% exact match, most other rows in the
0.05-0.25 range), but they're exactly the kind of case the error-propagation
study (`notes/tshivenda-error-propagation.md`) and Objective 6's reliability
threshold exist to quantify.

**Diacritics update**: ṋ now appears correctly in both refs and hyps (e.g.
"yaṋu", "nṋe", "ṋetshedza") - resolves the pilot v2 finding that ṋ was never
emitted (0 of 84 occurrences at that scale). The extra data and epoch fixed it.

### Re-measured error model (full-scale predictions)

From `results/preds_full/*.csv` via `ErrorModel.from_prediction_files`
(`src/error_propagation/corrupt_transcripts_ven.py`):

- S:D:I = **54.4 : 16.5 : 29.0** (pilot v2 was 64.4 : 32.7 : 2.9) - as the
  model got stronger, deletions fell further and, notably, **insertions
  jumped from near-zero to nearly a third of all errors** - a real shift in
  error character, not just magnitude. Consistent with a model confident
  enough to occasionally over-generate rather than only drop hard words.
- Top confusion pairs still mostly single-word splits/particle swaps
  (u -> hu/uri, nga -> ngauri, unga -> nga), same linguistic flavor as
  before, not random noise.
- This is the error model now used for the error-propagation study
  (`src/error_propagation/run_degradation_study_ven.py`) via
  `--error-model results/preds_full/final_nchlt_test.csv results/preds_full/final_anv_dev_test.csv`.

## Whisper pilot v2 (rescoped) - best pilot-scale result

`src/asr/pilot_finetune_whisper_mps_ven.py --resume-from results/whisper-ven-pilot/final
--include-anv --epochs 5 --learning-rate 5e-5 --train-clips 12000 --eval-clips 500`,
resumed from pilot v1's weights, 12,000 raw train clips (NCHLT + ANV, 7,325 kept
after the 10s cap), 500 raw eval clips (430 kept), 5 epochs, M4 MacBook (MPS),
~12.3h. Rescoped down from an aborted first attempt that tried the full
60,087-clip pool and projected to ~51h (see `results/logs/README.md`).

| Epoch | eval WER | eval CER |
|---|---|---|
| 1 | 0.296 | 0.078 |
| 2 | 0.245 | 0.062 |
| 3 | 0.206 | 0.055 |
| 4 | 0.184 | 0.050 |
| 5 (final) | **0.182** | **0.048** |

Steady, clean improvement every epoch - no collapse. Comparison (200-clip
NCHLT test set, seed 42):

| Model | WER | CER |
|---|---|---|
| Whisper Large v3 zero-shot | 1.108 | 0.763 |
| Wav2Vec2 pilot v2 (+ANV, 8 total epochs) | 0.332 | 0.074 |
| Whisper pilot v1 (3 epochs, NCHLT only) | 0.265 | 0.060 |
| **Whisper pilot v2 (rescoped, +ANV, resumed)** | **0.182** | **0.048** |

Best result of any pilot run so far - beats pilot v1 by nearly 1/3 in WER,
using the same scale of extra data (~7,300 clips) that took Wav2Vec2 from
0.614 to 0.332. Model checkpoint saved to `results/whisper-ven-pilot-v2/final`
(gitignored - re-run to regenerate). Full log:
`results/logs/whisper_pilot_v2_wer0182.log`.

## Whisper pilot v1 - works cleanly, best result so far

`src/asr/pilot_finetune_whisper_mps_ven.py`, whisper-small, 5,000 NCHLT train clips
(<= 10 s cap, 4,974 kept), 493 eval clips, 3 epochs, M4 MacBook (MPS).
Placeholder language token "sw" (Swahili) used since Whisper has no `<|ven|>`
token - see the module docstring for the reasoning. NOT the real Stage 1
(whisper-large-v3, full data, GPU) - same proof-of-recipe role as the
Wav2Vec2 pilots.

| Epoch | eval WER | eval CER |
|---|---|---|
| 1 | 0.428 | 0.110 |
| 2 | 0.281 | 0.064 |
| 3 (final) | **0.265** | **0.060** |

Steady, clean improvement every epoch - no collapse pattern (contrast with
AfriHuBERT below, where WER/CER froze bit-for-bit across every attempt).
Already beats Wav2Vec2 pilot v2 (WER 0.332, CER 0.074) despite fewer total
epochs (3 vs. 8) and no ANV data in the mix - Whisper's pretraining is a
much stronger starting point for this small a pilot. Full log:
`results/logs/whisper_pilot_v1_wer0265.log`.

## AfriHuBERT (third model attempt) - failed, total training collapse

Attempted `ajesujoba/AfriHuBERT` as the third distinct ASR model family
(Seani asked for 3 architecturally distinct models; Wav2Vec2 and Whisper are
the other two). Confirmed via direct check it covers Tshivenda (1240
language tags, `ven` included) and loads via standard `transformers`
(`HubertForCTC`) - no extra toolkit needed, unlike the other candidate
checked (ESPnet's XEUS, which does cover Tshivenda but is not
`transformers`-native).

**Result: fails to train.** Six systematic attempts (default hyperparameters,
2x/3x lower learning rate, unfrozen feature encoder, 3x longer warmup,
manual blank-bias correction, and a definitive 25-epoch run with early
stopping disabled) all collapse into the model predicting a single dominant
token for 100% of frames - first the CTC blank token, then (after the
blank-bias fix) the single most frequent character instead. Loss decreases
smoothly in every attempt while predictions never change, then visibly
plateaus - ruling out "just needs more epochs." Confirmed by direct
inspection of decoded predictions at every stage, not just inferred from a
frozen WER number. Full evidence trail: `results/logs/README.md` and the six
`results/logs/hubert_attempt*.log` files.

**Decision (given the timeline): proceed with 2 model families - Wav2Vec2
and Whisper.** AfriHuBERT stands as a documented, fully-evidenced failed
attempt at a third family rather than an open thread to keep pulling on.
`src/asr/pilot_finetune_hubert_mps.py` and `notebooks/asr/colab_hubert_ven.ipynb`
stay in the repo in case debugging resumes later (e.g. on a real GPU, in
case this is specific to the MPS/eager-attention fallback this machine
required), but nothing further is planned against the current timeline.

Message sent to Seani:

> Update on the third model - AfriHuBERT covers Tshivenda and is cheap to
> integrate, but it won't train: 6 different configurations all collapse
> into predicting a single repeated token no matter what we change (lower
> learning rate, unfreezing, longer warmup, a targeted fix for the collapse,
> and a 25-epoch patience test that ruled out "just needs more time"). Fully
> documented in the repo. Given the deadline, we're going with 2 model
> families (Wav2Vec2 + Whisper) and documenting AfriHuBERT as an
> attempted-but-failed third, with the full diagnostic trail as evidence of
> the work. Let me know if you'd rather we pursue a different third model
> instead (the only other one I found with confirmed Tshivenda coverage is
> ESPnet's XEUS, but it needs a separate toolkit outside our current
> pipeline, so it's a bigger time cost).

## Fourth model attempt: MMS (2026-08-25)

Seani's response (session, 2026-08-24): keep trying model families - as many
as reasonable - and a checkpoint not having confirmed Tshivenda coverage is
not disqualifying on its own; "find a way to make it work, be creative, be
innovative." This reframes the earlier 2-family decision: AfriHuBERT is still
ruled out (training-dynamics failure, not a coverage problem), but the door
is open for a fourth attempt rather than stopping at two.

Checked `facebook/mms-1b-all` (the 1,162-language adapter-fine-tuned MMS
checkpoint) - confirmed it does not list Venda. Picked
`facebook/mms-300m` instead: the self-supervised MMS *base* checkpoint,
pretrained (not fine-tuned) on ~500,000 hours across 1,400+ languages.
Architecturally identical to Wav2Vec2 XLS-R-300M (both load via
`Wav2Vec2ForCTC`), which is exactly why XLS-R already works for Tshivenda
despite not being specifically labeled with Venda either - the recipe is
"fine-tune a custom CTC tokenizer on a strong general acoustic backbone,"
not "the checkpoint already knows the language." Same logic applies to
Whisper, which has no `<|ven|>` token at all.

`src/asr/pilot_finetune_mms_mps_ven.py` is a near line-for-line copy of
`pilot_finetune_wav2vec2_mps_ven.py` with the base checkpoint swapped -
same tokenizer, collator, and training loop. Verified it loads and trains
end-to-end with a 5-clip/1-epoch smoke test (`Wav2Vec2ForCTC LOAD REPORT`
shows the same UNEXPECTED/MISSING key pattern as XLS-R: quantizer/projection
heads discarded, `lm_head` freshly initialized for our vocab - i.e. loading
cleanly as a plain CTC fine-tune, not erroring).

**Real pilot result: fails the same way AfriHuBERT did - total blank
collapse.** 5,000 raw NCHLT clips (4,974 kept after the 10s filter), 2
epochs (Seani's guidance: fewer epochs is better, so this pilot used 2 not
3). `eval_wer`/`eval_cer` = 0.971/0.961 after epoch 1, 0.998/0.877 after
epoch 2 - numbers that landed suspiciously close to AfriHuBERT's own frozen
0.9709/0.9614, which was itself the signature of predicting blank on every
frame. Confirmed by direct inspection, not just the WER score: loaded the
saved checkpoint and ran raw predictions on 5 training clips - **100% of
frames predict the pad/blank token on every single example**, decoding to
an empty string every time, identical failure mode to AfriHuBERT's first
four attempts.

This is a genuinely interesting result on its own terms: MMS-300m and
XLS-R-300M are the same `Wav2Vec2ForCTC` architecture and the same
contrastive self-supervised pretraining objective (unlike AfriHuBERT's
masked-cluster-prediction objective) - yet one collapses on Tshivenda and
the other doesn't. So "architecture family" isn't what predicts collapse;
something about the specific pretraining data/scale/initialization is.
Not yet root-caused further (no lr sweep, no blank-bias-disfavor attempt
run for MMS - unlike the six systematic AfriHuBERT attempts, this is one
data point so far).

**Decision (2026-08-25): treat this as a second confirmed collapse.** Not
re-running AfriHuBERT's full six-attempt mitigation sweep against MMS too -
one clean, directly-verified collapse (not just inferred from WER) is enough
signal given AfriHuBERT already tried the same mitigations (lower LR,
blank-bias disfavor) and neither saved it; no reason to expect a different
outcome here.

**XEUS (ESPnet), checked and not pursued.** The remaining candidate with
*confirmed* native Tshivenda coverage. Checked what integration would take:
a non-mainline ESPnet fork (`espnet @ git+.../wanchichen/espnet.git@ssl`,
not the standard package), the best available community fine-tuning repo
is marked work-in-progress with limited docs, and it states CUDA as a
prerequisite - no confirmed path to the cheap local-MPS-pilot-first step
that caught real mistakes twice already in this project (e.g. Whisper v2's
51-hour scoping issue). Given the timeline, decided not to pursue it -
same treatment as AfriHuBERT: identified, evaluated, documented as a
deliberate no rather than an open thread.

**Status after MMS/XEUS: 2 working (Wav2Vec2, Whisper), 2 ruled out with
evidence (AfriHuBERT, MMS - both fail identically via total CTC blank
collapse), 1 identified-but-not-attempted (XEUS, integration cost too high
for the timeline).** Superseded below - a fifth attempt followed.

## Fifth model attempt: w2v-BERT 2.0 (2026-08-25)

Picked `facebook/w2v-bert-2.0` as the next attempt: Conformer-based
(convolution + self-attention, not a plain transformer over CNN features
like Wav2Vec2/HuBERT), hybrid contrastive + masked-prediction pretraining
objective (neither of the two objectives already tried), on 4.5M hours
across 143+ languages - both a different architecture family and a much
larger, broader pretraining pool than XLS-R, MMS, or AfriHuBERT. No
confirmed Tshivenda coverage, same situation as XLS-R/Whisper/MMS - and HF's
own fine-tuning writeup for this exact checkpoint demonstrates the same
"language not in pretraining, fine-tune anyway" approach on Mongolian, a
direct precedent for what we're doing here.

Architecturally further from our existing scripts than MMS was: loads via
`Wav2Vec2BertForCTC` (not `Wav2Vec2ForCTC`), and takes precomputed log-mel
`input_features` via `SeamlessM4TFeatureExtractor` rather than raw-waveform
`input_values` via `Wav2Vec2FeatureExtractor` - closer to the Whisper pilot
script's data-prep pattern than the Wav2Vec2/MMS one. Still reuses the
existing `tokenizers/ven/` CTC tokenizer unchanged. `src/asr/pilot_finetune_w2vbert_mps_ven.py`
built from this hybrid pattern; two real bugs caught and fixed via a
5-clip/1-epoch smoke test before trusting it:

1. `model.freeze_feature_encoder()` doesn't exist on `Wav2Vec2BertForCTC` -
   makes sense once you know why: unlike Wav2Vec2/MMS, there's no
   raw-waveform CNN feature encoder to freeze in the first place, since the
   model consumes precomputed features. Removed the call.
2. (nothing else broke - the `Wav2Vec2BertForCTC LOAD REPORT` showed *zero*
   UNEXPECTED keys, only the expected `lm_head` MISSING - a cleaner load
   than XLS-R or MMS got, both of which discard quantizer/projection heads.)

**Real pilot result: also collapses.** 5,000 raw NCHLT clips (4,974 kept),
2 epochs. `eval_wer`/`eval_cer` = 0.9535/0.9329, bit-for-bit identical
between epoch 1 and epoch 2 - frozen, not just similar. Confirmed by direct
inspection: loaded the checkpoint, ran raw predictions on 5 training clips -
~99% of frames predict blank, but unlike MMS (which decoded to nothing) the
one non-blank frame decodes to the character `'n'`, so every sample outputs
just `'n'`. Same "collapse to whichever single class is easiest" pattern as
AfriHuBERT's attempt 5 (which collapsed to `'a'` after a blank-bias fix),
just landing on a different token.

**Emerging pattern: 3 of 4 non-Whisper CTC fine-tunes have now collapsed**
(AfriHuBERT, MMS, w2v-BERT) - only Wav2Vec2 XLS-R-300M hasn't. None of MMS
or w2v-BERT got any mitigation attempt (lower LR, blank-bias disfavor) -
those were only ever tried against AfriHuBERT. Decision: keep testing
architectures rather than mitigation-sweep the existing failures, per
direction to continue trying model families.

## Sixth model attempt: data2vec-audio (2026-08-25)

All three collapses so far pretrain on some form of discretized target:
AfriHuBERT (cluster IDs), MMS (quantized codebook via contrastive+diversity
loss), w2v-BERT (hybrid quantized-contrastive + masked prediction). Picked
`facebook/data2vec-audio-large` specifically to test whether discretization
itself is the common thread - its objective is regression onto continuous,
contextualized teacher representations (an EMA teacher network's own hidden
states), no discretization anywhere in the pretraining target.

Honest caveat up front: this checkpoint is pretrained on Librispeech only
(960h, English) - no multilingual pretraining at all, the weakest
cross-lingual transfer prior of anything tried so far (XLS-R/MMS/w2v-BERT
all had broad multilingual pretraining). No multilingual data2vec-audio
checkpoint was found to substitute. If this collapses too, it's still an
informative data point (rules out discretization as the *sole* explanation)
but the weaker prior needs to stay part of interpreting the result either
way.

Reuses raw-waveform `Wav2Vec2FeatureExtractor` + the existing
`tokenizers/ven/` tokenizer, unlike w2v-BERT's log-mel setup - closer to
the XLS-R/MMS pattern. `src/asr/pilot_finetune_data2vec_mps_ven.py` smoke-tested
clean on the first try (model loads with only `lm_head` MISSING, same
clean-load pattern as w2v-BERT; `freeze_feature_encoder()` works here,
unlike w2v-BERT, since data2vec-audio does have a raw-waveform CNN feature
encoder).

**Real pilot result: collapses too, identically to blank.** 5,000 raw
NCHLT clips (4,974 kept), 2 epochs. `eval_wer`/`eval_cer` = 0.9709/0.9614,
bit-for-bit identical between epoch 1 and epoch 2 - and identical to
AfriHuBERT's own original blank-collapse signature. Confirmed by direct
inspection: 100% of frames predict blank on every one of 5 checked training
clips, decoding to an empty string every time.

**This disproves the discretization hypothesis.** data2vec-audio has no
discretized pretraining target at all, and it collapsed exactly like the
three that do. **4 of 5 non-Whisper CTC fine-tunes have now collapsed**
(AfriHuBERT, MMS, w2v-BERT, data2vec-audio) across four different
architectures and four different pretraining objectives (masked-cluster,
contrastive-quantized, hybrid, continuous-regression) - only XLS-R-300M
hasn't. The pattern now points at the shared training recipe (lr 1e-4,
frozen feature encoder, this exact batch/warmup setup) rather than model
choice - XLS-R may simply be the one checkpoint that tolerates it.

**Recipe theory tested, disproven too.** Re-ran MMS at `--learning-rate 3e-5`
(3x lower, same reduction factor tried against AfriHuBERT) - identical
result: `eval_wer`/`eval_cer` = 0.9709/0.9614, frozen across both epochs,
100% blank on every checked frame (confirmed by direct inspection again,
not just WER). Log: `results/logs/mms_attempt2_lowlr_still_collapsed.log`.

**Where this leaves the model search.** Two explanations tried and disproven
in turn: not discretization (data2vec-audio has none, collapsed anyway), and
not the learning rate (3x lower didn't save MMS either). Wav2Vec2 XLS-R-300M
remains the only non-Whisper CTC checkpoint that trains cleanly on this
tokenizer/recipe, out of 5 tried, and nothing tested so far explains *why*
it's the exception rather than the rule. Further hyperparameter sweeps
(blank-bias disfavor, longer warmup, unfrozen feature encoder - the other
mitigations AfriHuBERT tried) remain untested against MMS/w2v-BERT/data2vec,
but two clean single-variable tests have now come back negative, so each
further sweep has lower expected payoff than it did before this round.

## Seventh model attempt: UniSpeech (2026-08-25)

Picked `microsoft/unispeech-large-1500h-cv`: multi-task pretraining
combining phonetically-aware contrastive self-supervision with supervised
phonetic CTC learning, on CommonVoice's multilingual pool. Notable
difference from every prior attempt: UniSpeech's own paper specifically
evaluates cross-lingual transfer to *unseen* languages via CommonVoice -
the exact scenario here, not an incidental side effect of broad pretraining
the way XLS-R/MMS/Whisper's Tshivenda transfer is.

Architecturally close to XLS-R (raw-waveform `Wav2Vec2FeatureExtractor`,
`UniSpeechForCTC`, `freeze_feature_encoder()` works) - smoke-tested clean
on the first try.

**Real pilot result: works.** No collapse. 5,000 raw NCHLT clips (4,974
kept), 2 epochs. `eval_wer`/`eval_cer` fell every epoch: 0.765/0.179 ->
**0.610/0.144** - comparable in shape to XLS-R's own first pilot
(0.614/0.151 at the same scale). Confirmed by direct inspection, not just
the WER score: decoded predictions on 5 training clips are genuinely
close to the references - one exact match ("ofisi ya muhasho wa zwa"),
the rest off by a word-boundary split or a single character, nothing like
the four collapses above. Log: `results/logs/unispeech_attempt1_works_wer0610.log`.

**UniSpeech is the second working non-Whisper model, after XLS-R.** Out of
6 non-Whisper checkpoints tried, 2 work (XLS-R, UniSpeech) and 4 collapse
(AfriHuBERT, MMS, w2v-BERT, data2vec-audio). Both working checkpoints share
a multi-task or discriminative element beyond pure self-supervision - XLS-R
via CTC-friendly contrastive pretraining validated at scale across 128
languages, UniSpeech via its explicit phonetic-CTC + contrastive multi-task
design, specifically built and validated for cross-lingual transfer. Worth
noting as a shape to the pattern, though not yet enough data points to call
it a confirmed rule.

## External model check: DSFSI's own multilingual Whisper (2026-08-25)

Per instruction to check for any model with confirmed Tshivenda support:
found `dsfsi-anv/za-anv-multilingual-whisper-v3-turbo` on Hugging Face -
`whisper-large-v3-turbo` fine-tuned on 7 South African languages including
Venda, trained on the same ANV (Swivuriso) corpus this project already
uses. Reported overall WER 0.1501 / CER 0.0510 (across all 7 languages,
not Tshivenda-isolated) - notably better than our own best pilot number,
though not a like-for-like comparison (different eval set, full-scale
training vs. our laptop-scale pilots).

**Could not evaluate it - the published repo is broken.** Attempted to
load it via `src/asr/zero_shot_baseline_ven.py` (which accepts any HF
model id); failed with a tokenizer construction error. Investigated
directly: the repo's `vocab.json` is 746 bytes - nowhere near a real
~50k-token Whisper vocabulary - and `merges.txt`/`tokenizer.json` are
missing entirely. Tried reconstructing a working tokenizer by combining
their `vocab.json` with the base `openai/whisper-large-v3-turbo`'s
`merges.txt`/`special_tokens_map.json`/`tokenizer_config.json` - still
failed (`Token \`Ġ\` out of vocabulary`), confirming their vocab.json
itself is incomplete/corrupted, not just missing companion files. This is
an upload problem on DSFSI's end, not something fixable from outside their
repo.

**Worth raising with Seani directly**: DSFSI is her own research group
(AfriDSAI/Data Science for Social Impact). She may be able to get a working
checkpoint or the real per-language (Tshivenda-isolated) numbers directly,
which would be a strong external benchmark for this whole ASR comparison -
a full-scale multilingual Whisper-v3-turbo trained on the exact same source
data we use, rather than another architecture bet.

## Pilot v2 update

v2: resumed from v1's weights, added ANV clips <= 10s to the mix (7,325 mixed
train clips after the length cap), 5 more epochs, lr 5e-5. 6.5 h on the M4.
Validation WER fell every epoch: 60.4 -> 53.5 -> 51.1 -> 49.0 -> 48.7%.

Fixed 200-clip test samples (seed 42), directly comparable across rows:

| Model | NCHLT WER | NCHLT CER | ANV WER | ANV CER |
|---|---|---|---|---|
| Whisper Large v3 zero-shot | 1.108 | 0.763 | 1.072 | 0.501 |
| Pilot v1 (5k NCHLT, 3 ep) | 0.614 | 0.151 | 0.851 | 0.257 |
| **Pilot v2 (+ANV, +5 ep, resumed)** | **0.332** | **0.074** | **0.537** | **0.127** |

- NCHLT: 33% WER / 7.4% CER - short read-speech clips are now frequently
  transcribed exactly.
- ANV gap closed substantially (85% -> 54% WER) after including spontaneous
  speech in training.
- **Diacritics emerged**: v1 emitted zero; v2 emits ḓ ḽ ṅ ṱ at roughly
  reference-level frequency (NCHLT: 58 in refs, 67 in hyps). Exception: ṋ is
  still never emitted (0 of 84 across both sets) - likely being confused with
  ṅ/n; worth checking after Stage 1.
- Re-measured error model: S:D:I = 64.4 : 32.7 : 2.9 (v1: 49.5 : 49.3 : 1.3) -
  as the model improves, deletions fall and substitutions dominate, close to
  the corruption engine's 60:25:15 default. Top confusions: nṋe->ne, unga->nga.
- Still a laptop pilot: 12% of available data, 10s clip cap. Stage 1 on GPU
  remains the real run.

### Reference vs. model output examples (Wav2Vec2 pilot v2, NCHLT test)

Real rows from `results/preds_pilot_v2/wav2vec2-final_nchlt_test.csv`, picked
to show the honest range at WER 0.332 / CER 0.074 - not cherry-picked to only
the best cases:

| Reference | Hypothesis |
|---|---|
| i fanela u dzhiela nzhele | i fanela u dzhiela nzhele *(exact match)* |
| ya u sumbedzwa tshirunzi na | ya u sumbedzwa tshirunzi na *(exact match)* |
| na vhuḓifhinduleli kha vhashumi nahone | na vhuḓifhenduleli kha vhashumi na hone |
| vhulimi zwine zwa khou bvelela | vhulimi zwine zwa khou bvelela *(exact match)* |
| havhudi vhune ha sa tou | havhuḓivhune ha sa to u |
| tsha kale musi vhasidzana vha | tshakale musi vha sidzana vha |

Errors cluster around word-boundary shifts and single-vowel/diacritic swaps
(vhuḓifhinduleli -> vhuḓifhenduleli, havhudi -> havhuḓivhune), not garbled or
unrelated output - consistent with the S:D:I error model measured above.

**Whisper's equivalent table is still pending** - the full-scale run
(`notes/whisper-full-scale-run-log.md`) has no saved predictions yet, and
generating them now would mean loading a second model into memory alongside
training while the machine is already swap-constrained (see that file's
21:15 entry - 23.9GB/24.6GB swap in use). Will add once the run finishes and
`evaluate_wav2vec2_ven.py`-style predictions are saved for Whisper (or via
`zero_shot_baseline_ven.py --model results/whisper-ven-pilot-v3/final
--save-predictions`, since that script's `pipeline()` call accepts a local
checkpoint path).

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
  nearly absent. Use via corrupt_transcripts_ven.py --error-model.
- Top confusion pairs are linguistically plausible: nṋe->ne, zwine->zine,
  hu<->u, wana->wa (diacritic loss and cluster simplification dominate).

## Caveats

- Pilot only: small subset, short-clip cap, single corpus, few epochs.
- Deletion-heavy ratio partly reflects degenerate merged-word output on long
  ANV clips; re-measure after Stage 1.
- Regenerate predictions: src/asr/evaluate_wav2vec2_ven.py --checkpoint
  results/wav2vec2-ven-pilot/final --save-predictions results/preds_pilot
