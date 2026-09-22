"""Shared audio augmentation for Tshivenda ASR fine-tuning (proposal Objective 2, §4.3).

Speed perturbation: resample-based tempo change (0.9x / 1.1x), the technique
(Biswas et al., 2022) validated on South African speech - changes duration
*and* pitch together (unlike a phase-vocoder time-stretch), simulating a
genuinely faster/slower speaker rather than the same speaker sped up.
Implemented with plain numpy interpolation so it needs no extra audio
dependency (this project's environment has neither librosa nor torchaudio).

SpecAugment is deliberately NOT implemented here - both Wav2Vec2Config and
WhisperConfig already support it natively (`apply_spec_augment`,
`mask_time_prob`, `mask_feature_prob`), so the pilot scripts enable it
directly on the model config rather than duplicating HuggingFace's own
paper-matching (Park et al., 2019) implementation.
"""

import numpy as np

SPEED_RATES = (0.9, 1.1)


def speed_perturb(audio: np.ndarray, rate: float) -> np.ndarray:
    """Resample `audio` by `rate` (0.9 = 10% slower, 1.1 = 10% faster)."""
    n_new = max(1, int(round(len(audio) / rate)))
    x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
    x_new = np.linspace(0, 1, num=n_new, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)
