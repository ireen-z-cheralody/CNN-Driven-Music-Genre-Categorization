import numpy as np
import matplotlib.pyplot as plt
import librosa.display

from config import GENRES

data = np.load("results/dataset_split.npz")
X_train = data["X_train"]
y_train = data["y_train"]

# grab the first training example and its genre label
example = X_train[0]
label = GENRES[y_train[0]]

plt.figure(figsize=(10, 4))
librosa.display.specshow(example, x_axis="time", y_axis="mel", sr=22050, hop_length=512)
plt.colorbar(format="%+2.0f")
plt.title(f"Mel Spectrogram — genre: {label}")
plt.tight_layout()
plt.savefig("results/example_spectrogram.png")
print("Saved to results/example_spectrogram.png")