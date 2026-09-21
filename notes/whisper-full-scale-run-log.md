# Whisper Full-Scale Run - Live Tracking

Live log for the local M4/MPS full-scale Whisper run, kept up to date as it
trains. Not a substitute for `notes/pilot-ven-results.md` (the pilot
write-up) - this file exists specifically to show progress *during* a long
run, gets folded into the pilot notes once it finishes.

## Config

- Script: `src/asr/pilot_finetune_whisper_mps_ven.py`
- Resumed from: `results/whisper-ven-pilot-v2/final` (best pilot so far, WER 0.182 / CER 0.048)
- Data: full NCHLT + ANV train pool (`--train-clips 100000`, clamped to the
  actual pool size - 60,087 raw clips, ~36,600 expected after the 10s MPS
  memory-safety filter), `--include-anv`
- Epochs: 1 (reduced from an initial 2-epoch plan, to keep total time closer
  to ~12-13h instead of ~25h - matches Seani's "fewer epochs" guidance)
- Learning rate: 5e-5 (same as pilot v2)
- Started: 2026-09-21, ~15:11 local time
- Log file: `/tmp/whisper_full_scale_v3.log`

## Why this run, and why this scope

Both Colab and Kaggle turned out to be too much friction to run reliably
(session resets, environment-detection edge cases, missing dependencies -
see the notebook fixes earlier in this branch). Decision: run the full-scale
Whisper training locally instead, since it's the stronger model (best pilot
result) and the flagship number for the report. Wav2Vec2 stays at its
validated pilot-scale number (WER 0.332) as supporting evidence rather than
also being pushed to full scale, to keep this tractable on one laptop.

Estimated time (~12-13h) is based on real measured throughput from the pilot
v2 run (1.21s/clip-epoch), not a guess - same discipline that caught the
original over-scoped Whisper v2 attempt earlier in this project (projected
~51h, killed after 1.4h once the real step-rate revealed the problem). Will
flag immediately here if the real step-rate looks off once training starts.

## Progress log

| Time | Status |
|---|---|
| 15:11 | Launched (1 epoch, resumed from pilot v2). Data mapping/filtering in progress (~60k raw clips). |

*(updated as the run progresses - see notes below on update cadence)*

## A note on update cadence

I can't wake myself up on a timer inside a normal chat session - I only get
to check on this run when you message me, or via the one-shot notification
when the whole process finishes (not incremental checkpoints along the way).
So "every hour automatically, without you asking" isn't something I can do
as a plain background habit in this session.

Two ways to actually get that:
1. **Just ping me periodically** ("how's it looking") - I'll pull the real
   log and update this file every time, same as I've been doing.
2. **Use `/loop`** (e.g. `/loop 90m`) if you want me to self-pace and check
   in on a timer without you having to prompt each time - that's the
   mechanism actually built for this.

Defaulting to (1) unless you want (2) set up instead.
