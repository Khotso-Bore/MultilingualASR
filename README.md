# Multilingual ASR for Misinformation Detection

COS700 research project: an end-to-end speech-to-misinformation-detection pipeline for **Sepedi**, **Setswana**, and **Tshivenda**. ASR transcriptions (Wav2Vec 2.0 / Whisper Large v3) are fed into multilingual classifiers (AfroXLM-RoBERTa / XLM-RoBERTa), with a systematic study of how transcription errors propagate into downstream misinformation classification failures.

See the full research proposal for background, research questions, and methodology.

## Team

| Name | Student Number |
|---|---|
| Khotso Bore | u19180642 |
| Mulisa Musehane | u21450162 |
| Khensani Chabalala | u23826305 |

Supervisors: Seani Rananga, Mahlatse Mbooi

## Repo structure

Each language sub-study gets its own scripts/notebooks under the same
pipeline-stage grouping; the Tshivenda track (the most complete so far) looks
like this:

```
dataset/                     # untouched, as originally set up
  get_dataset.py                # pulls audio metadata (NCHLT, African Next Voices) from HF into CSVs
  dsfsi-anv/                    # per-language NCHLT / African Next Voices metadata CSVs
  processed/                    # regeneratable preprocessed CSVs + audio (gitignored)

src/                          # reusable pipeline code, grouped by stage
  text_norm_ven.py                   # shared Tshivenda text normalisation - used across every stage below
  preprocessing/                 # NCHLT/ANV -> processed CSVs, CTC tokenizer
  asr/                            # ASR fine-tuning, evaluation, and shared augmentation
    audio_augment_ven.py               # speed perturbation (Objective 2) - numpy-only, no extra audio dependency
    pilot_finetune_whisper_mps_ven.py  # Whisper - the working flagship model; also handles LoRA/two-stage (Objective 3)
    pilot_finetune_wav2vec2_mps_ven.py # Wav2Vec2 - the working non-Whisper model; --model swaps to any other checkpoint (e.g. XLSR-53)
    pilot_finetune_hubert_mps_ven.py   # AfriHuBERT / SSA-HuBERT - both collapsed, kept for the debugging record
    pilot_finetune_mms_mps_ven.py      # collapsed
    pilot_finetune_w2vbert_mps_ven.py  # collapsed
    pilot_finetune_data2vec_mps_ven.py # collapsed
    pilot_finetune_unispeech_mps_ven.py # the second working non-Whisper model
    zero_shot_baseline_ven.py          # zero-shot Whisper baseline AND the standardized eval used for every final checkpoint
    evaluate_wav2vec2_ven.py           # standardized eval for any CTC checkpoint (despite the filename)
  classification/                # misinformation-classifier proxy dataset + training
  error_propagation/             # controlled-WER transcript corruption + the degradation study (§4.6, Objectives 5/6)
    corrupt_transcripts_ven.py         # the corruption engine + measured error-model support
    run_degradation_study_ven.py       # runs the classifier across WER levels, produces the WER-vs-F1 curve

notebooks/                    # thin wrappers around src/ for local iteration, plus Colab bootstrap notebooks
  notebook.ipynb                 # pre-language-split streaming-sample EDA
  za_next_voices_eda.ipynb       # pre-language-split fuller EDA + plots
  finetune_wav2vec2.ipynb        # Khotso's generic Setswana smoke-test template
  preprocessing/ asr/ classification/ error_propagation/   # mirrors src/ above

tokenizers/ven/                # committed custom CTC tokenizer (Tshivenda's 32-character set)

notes/                         # narrative write-ups of what was tried and why - read these first
  conclusions.md                  # start here: the actual answers to the Main RQ and Sub-questions 1-5, with evidence links
  pilot-ven-results.md            # the full ASR story - every model tried, full-scale results, augmentation, two-stage/LoRA
  whisper-full-scale-run-log.md   # live progress log kept during the Whisper/Wav2Vec2 full-scale training runs
  tshivenda-classifier-proxy.md   # misinformation-classifier proxy dataset rationale + results (Objective 4)
  tshivenda-error-propagation.md  # WER-vs-classification-accuracy study (Objectives 5/6)

results/
  logs/                          # tracked: raw output from every training run (success, failure, or abort)
    README.md                      # full run count and index - the evidence trail behind notes/
  *                              # everything else here is gitignored (model checkpoints, predictions - regeneratable)

requirements.txt
README.md
```

No dedicated `configs/` or `tests/` folder yet - add them the same way, alongside
what already exists rather than restructuring around them. A pass to
physically group these files into `src/asr/`, `src/preprocessing/` etc.
subfolders more finely (they're already grouped by top-level stage above,
but `src/asr/` itself has grown to 10 files) is planned but not yet done -
see the open items in `notes/conclusions.md`.

## Setup

Requirements:

- **Python 3.10+** (notebooks were authored against 3.12, anything 3.10 or newer should work)
- **ffmpeg** - needed by the `datasets` audio backend for decoding compressed formats (e.g. mp3). Install before doing anything with audio:
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt-get install ffmpeg`
  - Windows: `winget install ffmpeg` (or download from ffmpeg.org and add to PATH)
- **git** and **git-lfs** are not required yet, but install [git-lfs](https://git-lfs.com/) ahead of time if we start tracking any binary artefacts (e.g. small audio samples) directly in the repo.

Clone and set up a virtual environment:

```bash
git clone https://github.com/Khotso-Bore/MultilingualASR.git
cd MultilingualASR

python3 --version        # confirm 3.10+
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Register the environment as a Jupyter kernel so notebooks pick up the right packages:

```bash
python -m ipykernel install --user --name multilingual-asr --display-name "MultilingualASR"
jupyter notebook   # or: code . and open the notebook in VS Code, selecting the multilingual-asr kernel
```

The datasets used (`dsfsi-anv/multilingual-nchlt-dataset`, `dsfsi-anv/za-african-next-voices-compressed`) are public on the Hugging Face Hub, so no account is required to run `dataset/get_dataset.py` as-is. If we hit Hub rate limits, authenticate once with:

```bash
huggingface-cli login
```

Verify the setup works end to end:

```bash
python dataset/get_dataset.py
```

## Branching strategy

Two long-lived branches:

- **`main`** - final/stable. Only ever updated via pull request from `dev`. This is what you would hand in / demo.
- **`dev`** - integration branch. This is the day-to-day shared working state.

Topic branches are short-lived and branch off `dev`, not `main`.

### 1. Branch off `dev`

```bash
git checkout dev
git pull origin dev
git checkout -b <type>/<short-description>
```

Branch types:

| Prefix | Use for |
|---|---|
| `feature/` | New pipeline components (e.g. `feature/spec-augment`, `feature/asr-finetune-tsn`) |
| `fix/` | Bug fixes (e.g. `fix/nchlt-loader-encoding`) |
| `experiment/` | Exploratory work / model trials that may not land as-is (e.g. `experiment/xlsr53-vs-xlsr300m`) |
| `docs/` | Documentation only (e.g. `docs/readme-update`) |
| `data/` | Dataset acquisition/preprocessing changes (e.g. `data/swivuriso-vad`) |
| `refactor/` | Restructuring existing code/files without changing behaviour (e.g. `refactor/tshivenda-repo-structure`) |

Keep branches scoped to one piece of work.

### 2. Make your changes

Commit early and often with clear messages. Rebase on `dev` periodically if your branch lives for a while, to avoid drifting too far and hitting large merge conflicts later:

```bash
git fetch origin
git rebase origin/dev
```

### 3. Merge into `dev` via pull request

**A PR is required here too, even for small or obviously-correct changes -
nobody merges straight to `dev` themselves.** (This section used to say
otherwise; that turned out not to match how the team actually wants to
work, after a direct-to-`dev` merge needed a revert. Updated 2026-09-25.)

```bash
git push origin <type>/<short-description>
gh pr create --base dev --title "..." --body "..."
```

Leave the PR for the team to review and merge - don't self-merge, even
when the fix is uncontroversial.

### 4. Promote `dev` to `main` via pull request

Once `dev` is in a good, working state (e.g. a project phase is complete, or before a demo/submission), open a PR to bring it into `main`:

```bash
gh pr create --base main --head dev --title "..." --body "..."
```

PR description should cover: what changed since the last promotion, and how it was validated (e.g. WER/CER numbers, notebook output, screenshots of plots). At least one other team member reviews before merge. Merge (not squash) so `main`'s history reflects the `dev` commits it absorbed.
