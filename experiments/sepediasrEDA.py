# %% [markdown]
# # Sepedi ASR EDA — Whisper / Wav2Vec2 Benchmarking

# %%
# ---- 1. SETUP ----
import os
import json
import glob
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

DATA_DIR = Path("/Users/khensanichabalala/DSFSI/MultilingualASR/MultilingualASR/experiments/sepedi-asr-eda/dataset")  # e.g. Path("/Users/khensani/Documents/NCHLT_Sepedi")
OUT_DIR = Path("./eda_outputs")
OUT_DIR.mkdir(exist_ok=True)

# %%
# ---- 2. FOLDER INVENTORY ----
def inventory_folder(root, max_depth=3):
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        indent = "  " * depth
        print(f"{indent}{Path(dirpath).name}/  ({len(filenames)} files)")
        for f in filenames[:5]:
            print(f"{indent}  - {f}")
        if len(filenames) > 5:
            print(f"{indent}  ... and {len(filenames)-5} more")

inventory_folder(DATA_DIR)

exts = {}
for p in DATA_DIR.rglob("*"):
    if p.is_file():
        exts[p.suffix] = exts.get(p.suffix, 0) + 1
print("\nFile extension counts:", exts)

# %%
# ---- 3a. LOCATE TRANSCRIPTS + DTD AUTOMATICALLY (handles nested folder structures) ----
# transcriptions/ can be nested anywhere under DATA_DIR (e.g. DATA_DIR/audio/transcriptions/)
# so we search for it rather than hardcoding the path.
transcription_dirs = [p for p in DATA_DIR.rglob("*") if p.is_dir() and p.name == "transcriptions"]
if not transcription_dirs:
    raise FileNotFoundError("No folder named 'transcriptions' found anywhere under DATA_DIR — check the folder name/spelling.")
TRANSCRIPT_DIR = transcription_dirs[0]
print(f"Found transcriptions folder at: {TRANSCRIPT_DIR}")

train_xml = TRANSCRIPT_DIR / "nchlt_nso.trn.xml"
test_xml = TRANSCRIPT_DIR / "nchlt_nso.tst.xml"
print(f"train_xml exists: {train_xml.exists()}")
print(f"test_xml exists: {test_xml.exists()}")

# Print the DTD — this defines the exact tag/attribute names we need for parsing
for dtd_file in TRANSCRIPT_DIR.glob("*.dtd"):
    print(f"\n===== {dtd_file.name} =====")
    print(dtd_file.read_text(encoding="utf-8"))

# %%
# ---- 3b. PRINT XML TREE STRUCTURE (first few records) ----
import xml.etree.ElementTree as ET

def preview_xml_structure(xml_path, max_records=2):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    print(f"Root tag: <{root.tag}> attrib={root.attrib}")

    for i, child in enumerate(root):
        if i >= max_records:
            print(f"... ({len(root)} total top-level children)")
            break
        print(f"\n--- Record {i}: <{child.tag}> attrib={child.attrib} ---")
        for sub in child.iter():
            text_preview = (sub.text or "").strip()[:60]
            print(f"  <{sub.tag}> attrib={sub.attrib} text={text_preview!r}")

preview_xml_structure(train_xml, max_records=2)

# %%
# ---- 3c. PARSE XML INTO DATAFRAME (adjust tag/attribute names after viewing 3a/3b output) ----
def parse_nchlt_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []
    # PLACEHOLDER: adjust tag names below once you've seen the actual structure from 3b.
    # Common NCHLT pattern: root -> <recording audio="...wav"> -> <orth>transcript</orth>
    for rec in root.iter():
        audio_attr = rec.attrib.get("audio") or rec.attrib.get("filename") or rec.attrib.get("id")
        if audio_attr:
            # look for transcript text in this element or its children
            transcript = (rec.text or "").strip()
            if not transcript:
                for sub in rec:
                    if sub.text and sub.text.strip():
                        transcript = sub.text.strip()
                        break
            rows.append({"stem": Path(audio_attr).stem, "transcript": transcript})
    return pd.DataFrame(rows)

train_df = parse_nchlt_xml(train_xml)
print(train_df.shape)
train_df.head()

# %%
# ---- 3d. MATCH TRANSCRIPTS TO AUDIO FILES BY STEM ----
# transcriptions/ lives inside the audio folder, so its parent IS the audio root
AUDIO_DIR = TRANSCRIPT_DIR.parent
print(f"Using audio root: {AUDIO_DIR}")

audio_files = [p for p in AUDIO_DIR.rglob("*.wav")]
audio_by_stem = pd.DataFrame({"audio_path": [str(p) for p in audio_files],
                              "stem": [p.stem for p in audio_files]})

df = audio_by_stem.merge(train_df, on="stem", how="left")
print(f"Audio files: {len(audio_by_stem)}")
print(f"Matched transcripts: {df['transcript'].notna().sum()}")
print(f"Unmatched: {df['transcript'].isna().sum()}")
df.head()

# %%
# ---- 4. BASIC NULL CHECK + OPTIONAL SPEAKER ID FROM FILENAME ----
AUDIO_COL = "audio_path"
TEXT_COL = "transcript"
SPEAKER_COL = None  # set below if speaker ID can be parsed from the filename stem

print("Missing transcripts:", df[TEXT_COL].isna().sum())
print("Missing audio paths:", df[AUDIO_COL].isna().sum())

# Drop unmatched rows before running the rest of the EDA (keep a copy of the discarded ones first)
unmatched = df[df[TEXT_COL].isna()]
unmatched.to_csv(OUT_DIR / "unmatched_audio_files.csv", index=False)
df = df.dropna(subset=[TEXT_COL]).reset_index(drop=True)
print(f"Proceeding with {len(df)} matched rows.")

# NCHLT filenames often encode speaker ID as a prefix, e.g. "nso_001_utt023" -> speaker "nso_001"
# Uncomment and adjust the split logic below if this applies to your files:
SPEAKER_COL = "speaker_id"
df[SPEAKER_COL] = df["stem"].apply(lambda s: "_".join(s.split("_")[:2]))

# %%
# ---- 5. AUDIO DURATION + SAMPLE RATE + CORRUPT FILE CHECK ----
durations, sample_rates, rms_means, bad_files = [], [], [], []

for path in df[AUDIO_COL]:
    try:
        y, sr = librosa.load(path, sr=None)
        durations.append(librosa.get_duration(y=y, sr=sr))
        sample_rates.append(sr)
        rms_means.append(float(np.mean(librosa.feature.rms(y=y))))
    except Exception as e:
        durations.append(np.nan)
        sample_rates.append(np.nan)
        rms_means.append(np.nan)
        bad_files.append((path, str(e)))

df["duration_sec"] = durations
df["sample_rate"] = sample_rates
df["rms_mean"] = rms_means

print(f"Corrupt/unreadable files: {len(bad_files)}")
pd.DataFrame(bad_files, columns=["path", "error"]).to_csv(OUT_DIR / "corrupt_files.csv", index=False)

# %%
# ---- 6. DURATION DISTRIBUTION PLOT ----
plt.figure(figsize=(8, 4))
plt.hist(df["duration_sec"].dropna(), bins=40)
plt.axvline(0.5, color="red", linestyle="--", label="0.5s min")
plt.axvline(30, color="red", linestyle="--", label="30s max")
plt.title("Utterance Duration Distribution")
plt.xlabel("Seconds")
plt.ylabel("Count")
plt.legend()
plt.savefig(OUT_DIR / "duration_distribution.png", dpi=150)
plt.show()

print(df["duration_sec"].describe())
print("Outliers (<0.5s or >30s):", ((df["duration_sec"] < 0.5) | (df["duration_sec"] > 30)).sum())

# %%
# ---- 7. SAMPLE RATE CHECK ----
print(df["sample_rate"].value_counts())
# Both Whisper and Wav2Vec2 expect 16kHz — flag anything else for resampling

# %%
# ---- 8. SPEAKER BALANCE (only if SPEAKER_COL is set) ----
if SPEAKER_COL and SPEAKER_COL in df.columns:
    speaker_counts = df[SPEAKER_COL].value_counts()
    plt.figure(figsize=(10, 4))
    speaker_counts.plot(kind="bar")
    plt.title("Utterances per Speaker")
    plt.ylabel("Count")
    plt.savefig(OUT_DIR / "speaker_balance.png", dpi=150)
    plt.show()
    print(speaker_counts.describe())
else:
    print("No speaker column set — skipping speaker balance check.")

# %%
# ---- 9. RMS ENERGY DISTRIBUTION (near-silent / clipped clip detection) ----
plt.figure(figsize=(8, 4))
plt.hist(df["rms_mean"].dropna(), bins=40)
plt.title("Mean RMS Energy per Clip")
plt.xlabel("RMS")
plt.savefig(OUT_DIR / "rms_distribution.png", dpi=150)
plt.show()

low_energy_thresh = df["rms_mean"].quantile(0.01)
print(f"Suspiciously quiet clips (bottom 1%): {(df['rms_mean'] < low_energy_thresh).sum()}")

# %%
# ---- 10. TRANSCRIPT LENGTH STATS ----
df["word_count"] = df[TEXT_COL].fillna("").apply(lambda t: len(t.split()))
df["char_count"] = df[TEXT_COL].fillna("").apply(len)
df["words_per_sec"] = df["word_count"] / df["duration_sec"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["word_count"], bins=40)
axes[0].set_title("Word Count per Transcript")
axes[1].hist(df["words_per_sec"].replace([np.inf, -np.inf], np.nan).dropna(), bins=40)
axes[1].set_title("Words per Second (alignment sanity check)")
plt.savefig(OUT_DIR / "transcript_length_stats.png", dpi=150)
plt.show()

print(df[["word_count", "char_count", "words_per_sec"]].describe())

# %%
# ---- 11. VOCABULARY SIZE + TOP TOKENS ----
from collections import Counter

all_tokens = " ".join(df[TEXT_COL].fillna("")).split()
vocab = Counter(all_tokens)
print(f"Total tokens: {len(all_tokens)}")
print(f"Unique vocabulary size: {len(vocab)}")
print(f"Type-token ratio: {len(vocab) / len(all_tokens):.4f}")
print("\nTop 20 most frequent tokens:")
for word, count in vocab.most_common(20):
    print(f"  {word}: {count}")

pd.DataFrame(vocab.most_common(), columns=["token", "count"]).to_csv(
    OUT_DIR / "vocab_frequency.csv", index=False
)

# %%
# ---- 12. OOV RISK vs WHISPER TOKENIZER ----
# Requires: pip install transformers
from transformers import WhisperTokenizer

whisper_tok = WhisperTokenizer.from_pretrained("openai/whisper-small", language="sw", task="transcribe")
# ^ Whisper has no native Sepedi tag; using a related/multilingual tag as a proxy is expected —
#   note this explicitly in your write-up since it motivates the zero-shot vs fine-tuned comparison

sample_fragmentation = []
for word in list(vocab.keys())[:2000]:  # sample for speed
    n_subtokens = len(whisper_tok.tokenize(word))
    sample_fragmentation.append(n_subtokens)

print(f"Mean subtokens per word (sampled): {np.mean(sample_fragmentation):.2f}")
print(f"Words needing >2 subtokens: {sum(1 for n in sample_fragmentation if n > 2)} / {len(sample_fragmentation)}")

# %%
# ---- 13. SPEAKER LEAKAGE CHECK ACROSS SPLITS (only if you have a 'split' column) ----
SPLIT_COL = None  # set to e.g. "split" if present

if SPLIT_COL and SPEAKER_COL and SPLIT_COL in df.columns:
    split_speakers = df.groupby(SPLIT_COL)[SPEAKER_COL].apply(set)
    splits = list(split_speakers.index)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = split_speakers[splits[i]] & split_speakers[splits[j]]
            print(f"Speaker overlap between {splits[i]} and {splits[j]}: {len(overlap)}")
else:
    print("No split/speaker columns set — skipping leakage check.")

# %%
# ---- 14. SAVE CLEANED METADATA FOR BASELINE SCRIPT ----
df.to_csv(OUT_DIR / "eda_enriched_metadata.csv", index=False)
print(f"Saved enriched metadata with {len(df)} rows to {OUT_DIR / 'eda_enriched_metadata.csv'}")

# %%
# ---- 15. BASELINE SCAFFOLD: ZERO-SHOT WER/CER (Whisper + Wav2Vec2) ----
# Requires: pip install jiwer transformers torch
print(">>> RUNNING UPDATED CELL 15")
import torch
from transformers import pipeline
import jiwer

DEVICE = 0 if torch.cuda.is_available() else -1

whisper_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=DEVICE)
w2v_pipe = pipeline("automatic-speech-recognition", model="dsfsi/w2v-bert-2.0-lwazi", device=DEVICE)

results = []
sample_df = df.dropna(subset=[TEXT_COL]).sample(min(50, len(df)), random_state=42)  # small dev subset first

for _, row in sample_df.iterrows():
    ref = row[TEXT_COL]
    try:
        # Load audio ourselves via librosa/soundfile instead of passing a file path directly —
        # avoids the transformers ASR pipeline's torchcodec/FFmpeg dependency, which can fail
        # to load on some macOS setups due to broken library rpaths.
        audio_array, sr = librosa.load(row[AUDIO_COL], sr=16000)  # both models expect 16kHz

        # IMPORTANT: transformers' ASR pipeline mutates (pops keys from) the dict you pass it,
        # so we must build a *fresh* dict for each pipe() call rather than reusing one —
        # reusing one means the second call gets an already-emptied dict and raises
        # "the dict needs to contain a 'raw' key... and a 'sampling_rate' key".
        whisper_hyp = whisper_pipe({"raw": audio_array, "sampling_rate": sr})["text"]

        # Diagnostic: what language does Whisper *think* it's hearing? Sepedi isn't in
        # Whisper's supported language list, so this tells us what it's defaulting to.
        detected_lang = None
        try:
            inputs = whisper_pipe.feature_extractor(audio_array, sampling_rate=sr, return_tensors="pt")
            lang_token_ids, lang_probs = whisper_pipe.model.detect_language(
                input_features=inputs.input_features
            )
            detected_lang = whisper_pipe.tokenizer.decode([lang_token_ids[0]]).strip("<|>")
        except Exception:
            pass  # not critical — WER/CER still get computed either way

        w2v_hyp = w2v_pipe({"raw": audio_array, "sampling_rate": sr})["text"]

        results.append({
            "audio_path": row[AUDIO_COL],
            "reference": ref,
            "whisper_hyp": whisper_hyp,
            "whisper_detected_lang": detected_lang,
            "wav2vec_hyp": w2v_hyp,
            "whisper_wer": jiwer.wer(ref, whisper_hyp),
            "whisper_cer": jiwer.cer(ref, whisper_hyp),
            "wav2vec_wer": jiwer.wer(ref, w2v_hyp),
            "wav2vec_cer": jiwer.cer(ref, w2v_hyp),
        })
    except Exception as e:
        print(f"Failed on {row[AUDIO_COL]}: {e}")

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "baseline_results.csv", index=False)

if len(results_df) == 0:
    print("WARNING: no utterances succeeded — check the error messages above before proceeding.")
else:
    print(f"Successfully processed {len(results_df)} / {len(sample_df)} utterances.")
    print("Mean WER — Whisper:", results_df["whisper_wer"].mean())
    print("Mean WER — Wav2Vec2:", results_df["wav2vec_wer"].mean())
    print("Mean CER — Whisper:", results_df["whisper_cer"].mean())
    print("Mean CER — Wav2Vec2:", results_df["wav2vec_cer"].mean())
    if "whisper_detected_lang" in results_df.columns:
        print("\nWhisper detected-language distribution (diagnostic — Sepedi isn't in Whisper's language list):")
        print(results_df["whisper_detected_lang"].value_counts())
# %%
