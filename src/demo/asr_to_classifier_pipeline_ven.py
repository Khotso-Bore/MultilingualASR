"""End-to-end demo: real Tshivenda audio -> Whisper transcript -> misinformation verdict.

Everywhere else in this repo, the ASR model and the misinformation classifier
are connected only through a simulation: notes/tshivenda-error-propagation.md
corrupts clean article text by a measured error rate and checks how the
classifier reacts. This script is the first place real audio actually flows
through both models to produce one verdict.

Read this before trusting the verdict:
The classifier (results/classifier/final) was trained on long Vukuzenzele
government articles, real ones plus fact-distorted synthetic fakes (see
notes/tshivenda-classifier-proxy.md). The NCHLT/ANV audio clips this script
transcribes are short single-sentence speech-corpus recordings, not
recordings of those articles - no audio of the actual real/fake proxy
content exists, and there is no Tshivenda text-to-speech model to make any
(checked: no facebook/mms-tts-ven on HuggingFace). So a verdict on an NCHLT
clip proves the two models mechanically chain together and gives a real
label + confidence, but it is not a scored "correct/incorrect" prediction -
there is no ground truth for whether a random read sentence is "real" or
"fake" misinformation. The only accuracy number that means anything is
Objective 4's 0.562, on the text-only proxy dataset.

Usage:
    python src/demo/asr_to_classifier_pipeline_ven.py --audio path/to/clip.wav
    python src/demo/asr_to_classifier_pipeline_ven.py --random-clip           # picks a real NCHLT/ANV test clip
    python src/demo/asr_to_classifier_pipeline_ven.py --random-clip --seed 7
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import soundfile as sf
import torch
from jiwer import wer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text_norm_ven import normalize_transcript

REPO_ROOT = Path(__file__).resolve().parents[2]
WHISPER_CHECKPOINT = REPO_ROOT / "results" / "whisper-ven-pilot-v2" / "final"
CLASSIFIER_CHECKPOINT = REPO_ROOT / "results" / "classifier" / "final"
EVAL_SETS = {
    "nchlt_test": REPO_ROOT / "dataset" / "processed" / "nchlt_ven" / "test.csv",
    "anv_dev_test": REPO_ROOT / "dataset" / "processed" / "anv_ven" / "dev_test.csv",
}
# build_misinfo_proxy_ven.py convention: label 1 = real, label 0 = fake.
# The saved checkpoint has no id2label mapping (plain num_labels=2 head), so
# this is hardcoded from the training script rather than read off the model.
LABEL_NAMES = {0: "fake", 1: "real"}


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_random_clip(seed):
    corpus = random.Random(seed).choice(list(EVAL_SETS.keys()))
    csv_path = EVAL_SETS[corpus]
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    row = random.Random(seed).choice(rows)
    return corpus, row["audio"], row["transcript"]


def transcribe(audio_path, device):
    asr = pipeline("automatic-speech-recognition", model=str(WHISPER_CHECKPOINT), device=device)
    audio, sr = sf.read(audio_path)
    raw = asr({"array": audio, "sampling_rate": sr})["text"]
    return normalize_transcript(raw)


def classify(text, device):
    tokenizer = AutoTokenizer.from_pretrained(str(CLASSIFIER_CHECKPOINT))
    model = AutoModelForSequenceClassification.from_pretrained(str(CLASSIFIER_CHECKPOINT))
    model.to(device).eval()

    inputs = tokenizer(text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    label = int(torch.argmax(probs).item())
    return LABEL_NAMES[label], probs[label].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", help="path to a Tshivenda .wav clip")
    parser.add_argument("--random-clip", action="store_true",
                        help="pick a real clip from the NCHLT test / ANV dev_test sets instead")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.audio and not args.random_clip:
        parser.error("pass --audio <path> or --random-clip")

    device = pick_device()

    reference = None
    if args.random_clip:
        corpus, audio_path, reference = pick_random_clip(args.seed)
        print(f"picked a real clip from {corpus}: {audio_path}")
    else:
        audio_path = args.audio

    print(f"\ntranscribing with {WHISPER_CHECKPOINT} (device: {device})...")
    hypothesis = transcribe(audio_path, device)
    print(f"transcript: {hypothesis}")
    if reference:
        ref_norm = normalize_transcript(reference)
        clip_wer = wer(ref_norm, hypothesis) if ref_norm else float("nan")
        print(f"reference:  {ref_norm}")
        print(f"WER on this clip: {clip_wer:.3f}")

    print(f"\nclassifying with {CLASSIFIER_CHECKPOINT} (device: {device})...")
    label, confidence = classify(hypothesis, device)
    print(f"verdict: {label} (confidence {confidence:.3f})")

    print("\nnote: this verdict has no ground truth to check against (see module "
          "docstring) - it shows the pipeline runs end to end on real audio, not a "
          "scored misinformation-detection accuracy.")
