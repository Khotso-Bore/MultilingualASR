"""Shared transcript normalisation for the Tshivenda ASR pipeline.

Extends the fine-tuning notebook's CHARS_TO_IGNORE with the backslash and
non-breaking space (\xa0) found in the ANV/Swivuriso transcripts (see the
Tshivenda EDA findings in notebooks/za_next_voices_eda.ipynb).

Tshivenda diacritics (ḓ ḽ ṅ ṋ ṱ) are preserved - they must end up in the
CTC tokenizer vocabulary.
"""

import re

# Notebook's CHARS_TO_IGNORE + backslash; \xa0 is handled as whitespace below
CHARS_TO_IGNORE = r'[,\?\.!\-\;\:"“%‘”�0-9\[\]\'\_\\]'


def normalize_transcript(text):
    """Lowercase, strip ignorable punctuation/digits, collapse whitespace."""
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(CHARS_TO_IGNORE, "", text)
    # \xa0 and any other whitespace variant -> single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()
