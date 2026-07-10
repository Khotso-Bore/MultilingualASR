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

```
dataset/                    # untouched, as originally set up
  get_dataset.py              # pulls audio metadata (NCHLT, African Next Voices) from HF into CSVs
  dsfsi-anv/
    nchlt/                      # per-language NCHLT metadata CSVs
    anv/                        # per-language African Next Voices metadata CSVs
notebooks/                  # exploratory notebooks
  notebook.ipynb               # streaming-sample EDA pattern
  za_next_voices_eda.ipynb     # fuller EDA + plots
requirements.txt
README.md
```

As new pipeline stages start, add a `src/` folder for reusable pipeline code (e.g. `src/asr/` for Wav2Vec 2.0 / Whisper fine-tuning, `src/classification/` for the misinformation classifiers, `src/error_propagation/` for the WER degradation analysis) and a `configs/` folder for per-experiment configs. `dataset/` stays as-is; new code and data folders get added alongside it rather than renaming or moving what is already there. No dedicated `tests/` folder for now. Experiment outputs (logs, per-run metrics, plots) stay local and gitignored (`results/`); only pull specific numbers into the report or a committed summary when they matter.

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
- **`dev`** - integration branch. Everyone's topic branches merge into `dev` directly (no PR needed for that step). This is the day-to-day shared working state.

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

Keep branches scoped to one piece of work.

### 2. Make your changes

Commit early and often with clear messages. Rebase on `dev` periodically if your branch lives for a while, to avoid drifting too far and hitting large merge conflicts later:

```bash
git fetch origin
git rebase origin/dev
```

### 3. Merge into `dev`

No PR required for this step, merge (or fast-forward) your topic branch back into `dev` yourself once it works:

```bash
git checkout dev
git pull origin dev
git merge <type>/<short-description>
git push origin dev
```

Still worth a quick heads-up to the team (e.g. in chat) for anything touching shared code, so you do not clash with someone else's in-flight branch.

### 4. Promote `dev` to `main` via pull request

Once `dev` is in a good, working state (e.g. a project phase is complete, or before a demo/submission), open a PR to bring it into `main`:

```bash
gh pr create --base main --head dev --title "..." --body "..."
```

PR description should cover: what changed since the last promotion, and how it was validated (e.g. WER/CER numbers, notebook output, screenshots of plots). At least one other team member reviews before merge. Merge (not squash) so `main`'s history reflects the `dev` commits it absorbed.
