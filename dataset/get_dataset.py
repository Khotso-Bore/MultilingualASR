from datasets import load_dataset, DatasetDict
import pandas as pd
import numpy as np
import os


subset_dict = {
    "Afrikaans": "afr",
    "isiNdebele": "nbl",
    "isiXhosa": "xho",
    "isiZulu": "zul",
    "Sepedi": "nso",
    "Sesotho": "sot",
    "Setswana": "tsn",
    "Siswati": "ssw",
    "Tshivenda": "ven",
    "Xitsonga": "tso",
}

# ── 2. Batched map function — operates on dicts of lists ──────────────────────
def extract_audio_meta_batched(batch):
    """
    In batched mode, `batch` is a dict where each value is a list.
    batch["audio"] is a list of audio dicts (array, sampling_rate, path).
    We add new keys and remove the raw audio — all in one pass.
    """
    paths, sample_rates, num_samples_list = [], [], []
    durations, sizes, channels = [], [], []

    for audio in batch["audio"]:
        if audio is None:
            paths.append(None); sample_rates.append(None)
            num_samples_list.append(None); durations.append(None)
            sizes.append(None); channels.append(None)
            continue

        samples    = audio.get_all_samples()
        array      = samples.data.numpy()
        sr         = int(samples.sample_rate)
        path       = getattr(audio, "path", None)
        n_samples  = array.shape[-1]
        n_channels = array.shape[0] if array.ndim > 1 else 1
        duration   = round(n_samples / sr, 4) if sr else None
        size_bytes = array.nbytes

        paths.append(path)
        sample_rates.append(sr)
        num_samples_list.append(n_samples)
        durations.append(duration)
        sizes.append(size_bytes)
        channels.append(n_channels)

    batch["audio_sampling_rate"] = sample_rates
    batch["audio_num_samples"]   = num_samples_list
    batch["audio_duration_s"]    = durations
    batch["audio_size_bytes"]    = sizes
    batch["audio_num_channels"]  = channels

    # Drop the raw audio column — this is the key OOM-prevention step
    del batch["audio"]

    return batch

# for subset_name, lang_code in subset_dict.items():
#     subset_data = load_dataset("dsfsi-anv/za-african-next-voices-compressed", lang_code)
#     print(f"Loaded {subset_name} dataset with {len(subset_data)} samples.")
#     # subset_data.save_to_disk(f"E:/datasets/dsfsi-anv/za-african-next-voices-compressed_{lang_code}")
#     # print("succesfull saved to disk")

#     # ── 3. Apply in batches (tune batch_size to your RAM) ─────────────────────────
#     ds_meta = subset_data.map(
#         extract_audio_meta_batched,
#         batched=True,
#         batch_size=50,          # lower if you still hit memory pressure
#         remove_columns=["audio"], # belt-and-suspenders: ensures HF drops it too
#         num_proc=4,              # ← number of parallel processes
#     )

#     # ── 4. Export to CSV ───────────────────────────────────────────────────────────
#     df = ds_meta.to_pandas()
#     df.to_csv(f"../dsfsi-anv/za-african-next-voices-compressed_{lang_code}.csv", index=False)
#     print(f"{lang_code} Done: {len(df)} rows × {len(df.columns)} cols")


if __name__ == "__main__":        # ← required on Windows
    
    for subset_name, lang_code in subset_dict.items():
        subset_data = load_dataset("dsfsi-anv/multilingual-nchlt-dataset", lang_code)
        # subset_data = DatasetDict({split: dataset.select(range(300)) for split, dataset in subset_data.items()})
        # print(subset_data)
        # print(subset_data.column_names)
        # break
        print(f"Loaded {subset_name} dataset with {len(subset_data)} samples.")
        # subset_data.save_to_disk(f"E:/datasets/dsfsi-anv/multilingual-nchlt-dataset_{lang_code}")
        # print(f"succesfully saved {subset_name} to disk")

        # ── 3. Apply in batches (tune batch_size to your RAM) ─────────────────────────
        ds_meta = subset_data.map(
            extract_audio_meta_batched,
            batched=True,
            batch_size=100,          # lower if you still hit memory pressure
            remove_columns=["audio"], # belt-and-suspenders: ensures HF drops it too
            num_proc=6,              # ← number of parallel processes
        )

        # ── 4. Export to CSV ───────────────────────────────────────────────────────────
        dfs = []
        for split, dataset in ds_meta.items():
            df = dataset.to_pandas()
            df["split"] = split
            dfs.append(df)
        
        all_dfs = pd.concat(dfs, ignore_index=True)
        all_dfs.to_csv(f"dsfsi-anv/multilingual-nchlt-dataset_{lang_code}.csv", index=False)
        #df = ds_meta.to_pandas()
        #df.to_csv(f"dsfsi-anv/za-african-next-voices-compressed_{lang_code}.csv", index=False)
        print(f"{subset_name} Done: {len(all_dfs)} rows × {len(all_dfs.columns)} cols")
