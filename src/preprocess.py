import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from collections import Counter

from config import GENRES, SR, N_MELS, N_FFT, HOP_LENGTH, SAMPLES_PER_TRACK

GTZAN_DIR = "data/gtzan/genres_original"
FMA_METADATA = "data/fma/fma_metadata/tracks.csv"
FMA_AUDIO_DIR = "data/fma/fma_medium"

MAPPING = {
    "Rock": "rock",
    "Hip-Hop": "hiphop",
    "Pop": "pop",
    "Classical": "classical",
    "Jazz": "jazz",
    "Country": "country",
    "Blues": "blues",
}

CAP_PER_GENRE = 500  # balancing cap; adjust after seeing real counts


# ---------- loading raw audio (no spectrograms yet) ----------

def fix_length(y):
    if len(y) < SAMPLES_PER_TRACK:
        y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)))
    else:
        y = y[:SAMPLES_PER_TRACK]
    return y


def load_gtzan(raw_audio, y):
    for genre in GENRES:
        folder = os.path.join(GTZAN_DIR, genre)
        if not os.path.isdir(folder):
            print(f"GTZAN folder missing for {genre}, skipping")
            continue
        label_idx = GENRES.index(genre)
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith(".wav"):
                continue
            try:
                audio, sr = librosa.load(os.path.join(folder, fname), sr=SR)
                raw_audio.append(fix_length(audio))
                y.append(label_idx)
            except Exception as e:
                print("skipped", fname, e)
        print(f"[GTZAN] {genre} done")


def load_fma(raw_audio, y):
    tracks = pd.read_csv(FMA_METADATA, index_col=0, header=[0, 1])
    medium = tracks[tracks[('set', 'subset')] <= 'medium']
    fma_genres = medium[('track', 'genre_top')].dropna()

    for track_id, genre in fma_genres.items():
        if genre not in MAPPING:
            continue
        genre_name = MAPPING[genre]
        label_idx = GENRES.index(genre_name)
        fpath = os.path.join(FMA_AUDIO_DIR, f"{track_id // 1000:03d}", f"{track_id:06d}.mp3")
        if not os.path.exists(fpath):
            continue
        try:
            audio, sr = librosa.load(fpath, sr=SR)
            raw_audio.append(fix_length(audio))
            y.append(label_idx)
        except Exception as e:
            print("skipped", track_id, e)
    print("[FMA] done")


# ---------- balancing (on raw audio, before splitting) ----------

def balance(raw_audio, y):
    y = np.array(y)
    counts = Counter(y)
    print("Counts before balancing:", {GENRES[k]: v for k, v in counts.items()})

    balanced_idx = []
    for genre_idx in set(y):
        idxs = np.where(y == genre_idx)[0]
        n = min(len(idxs), CAP_PER_GENRE)
        balanced_idx.extend(np.random.choice(idxs, n, replace=False))

    balanced_idx = np.array(balanced_idx)
    raw_bal = [raw_audio[i] for i in balanced_idx]
    y_bal = y[balanced_idx]
    print("Counts after balancing:", Counter(y_bal))
    return raw_bal, y_bal


# ---------- augmentation (training clips only) ----------

def augment(y, sr=SR):
    choice = np.random.choice(["stretch", "pitch", "noise"])
    if choice == "stretch":
        y_aug = librosa.effects.time_stretch(y, rate=np.random.uniform(0.9, 1.1))
        y_aug = fix_length(y_aug)  # stretching changes length, re-pad/trim
    elif choice == "pitch":
        y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.uniform(-2, 2))
    else:
        y_aug = y + np.random.normal(0, 0.005, y.shape)
    return y_aug


def augment_training_set(raw_train, y_train):
    """Adds one augmented copy per training clip, expanding training diversity."""
    aug_audio, aug_labels = [], []
    for audio, label in zip(raw_train, y_train):
        aug_audio.append(augment(audio))
        aug_labels.append(label)
    print(f"Added {len(aug_audio)} augmented clips to training set")
    return raw_train + aug_audio, np.concatenate([y_train, np.array(aug_labels)])


# ---------- spectrogram conversion (final step, on everything) ----------

def audio_to_melspec(y, sr=SR):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    S_db = librosa.power_to_db(S, ref=np.max)
    return ((S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)).astype(np.float32)


def to_spectrograms(raw_audio_list):
    return np.array([audio_to_melspec(y) for y in raw_audio_list])


def main():
    raw_audio, y = [], []
    load_gtzan(raw_audio, y)
    load_fma(raw_audio, y)

    raw_audio, y = balance(raw_audio, y)

    # split on raw audio, BEFORE spectrogram conversion
    idx = np.arange(len(raw_audio))
    idx_train, idx_temp, y_train, y_temp = train_test_split(idx, y, test_size=0.30, stratify=y, random_state=42)
    idx_val, idx_test, y_val, y_test = train_test_split(idx_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    raw_train = [raw_audio[i] for i in idx_train]
    raw_val = [raw_audio[i] for i in idx_val]
    raw_test = [raw_audio[i] for i in idx_test]

    # augment training clips only
    raw_train, y_train = augment_training_set(raw_train, y_train)

    print("Converting to mel spectrograms...")
    X_train = to_spectrograms(raw_train)
    X_val = to_spectrograms(raw_val)
    X_test = to_spectrograms(raw_test)

    os.makedirs("results", exist_ok=True)
    np.savez_compressed(
        "results/dataset_split.npz",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )
    print("Saved dataset_split.npz")
    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)


if __name__ == "__main__":
    main()