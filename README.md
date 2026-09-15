# Music Genre Categorization with CNN

A deep learning project that classifies music into seven genres using **Convolutional Neural Networks (CNNs)** trained on mel spectrograms generated from audio recordings.

## Overview

Manually organizing large music libraries by genre is slow, expensive, and subjective. This project explores an automated approach by training a deep learning model to predict a song's genre directly from its audio.

Rather than feeding raw audio into the model, each 30-second audio clip is converted into a **mel spectrogram**, which represents how the frequencies of a sound change over time. The spectrogram can then be treated similarly to an image, allowing a CNN to learn visual patterns associated with different musical genres.

The project compares the CNN against a traditional **Support Vector Machine (SVM)** baseline and evaluates the final model on both held-out test data and completely unseen music from external sources.

## Genres

The final model classifies music into seven genres shared between the two training datasets:

- Blues
- Classical
- Country
- Hip Hop
- Jazz
- Pop
- Rock

## Dataset

The project combines two publicly available music datasets:

- **GTZAN Genre Collection** — 1,000 audio clips across 10 genres
- **Free Music Archive (FMA) Medium** — approximately 25,000 tracks across 16 genres

Because the two datasets use different genre classifications, only the seven genres shared between them were retained. Overrepresented classes were downsampled to improve class balance.

### Audio Preprocessing

Each audio track was processed using `librosa`:

1. Resampled to **22,050 Hz**
2. Trimmed to **30 seconds**
3. Converted into a **128-band mel spectrogram**
4. Converted to decibel scale
5. Normalized
6. Saved as a NumPy array

Training data was augmented using:

- Time stretching
- Pitch shifting
- Additive Gaussian white noise

The final dataset was divided approximately into:

| Split | Samples |
|---|---:|
| Training | 3,808 |
| Validation | 408 |
| Test | 409 |
| **Total** | **4,625** |



## Model Architecture

The final CNN receives a **128 × 640 normalized mel spectrogram** as input.

The network consists of four convolutional blocks with progressively increasing channel depth:

```text
Input
  │
  ▼
Conv Block 1 — 32 channels
  │
  ▼
Conv Block 2 — 64 channels
  │
  ▼
Conv Block 3 — 128 channels
  │
  ▼
Conv Block 4 — 256 channels
  │
  ▼
Global Average Pooling
  │
  ▼
Fully Connected — 64 units
  │
  ▼
Fully Connected — 7 outputs
  │
  ▼
Genre Prediction
```

Each convolutional block contains:

- 3 × 3 convolution
- Batch normalization
- ReLU activation
- 2 × 2 max pooling
- Dropout

Instead of flattening the final feature maps, the model uses **Global Average Pooling**, reducing the final 256 feature maps to a 256-dimensional vector while significantly reducing the number of parameters.

The final fully connected layers map these features to the seven genre classes. The complete network contains approximately **616,807 trainable parameters**.

## Baseline: Support Vector Machine

An **SVM** was implemented as a traditional machine learning baseline.

Unlike the CNN, which learns features directly from mel spectrograms, the SVM uses numerical audio features extracted with `librosa`. These features summarize properties such as:

- Texture
- Rhythm
- Pitch

The SVM achieved:

**79.17% test accuracy**

This provides a useful benchmark for determining whether the additional complexity of the CNN improves genre classification.

## Results

### CNN Performance

The final CNN achieved:

| Metric | Result |
|---|---:|
| Test Accuracy | **79.46%** |
| Validation Accuracy | **76.4%** |
| Macro F1 | **0.77** |
| Weighted F1 | **0.79** |
| Trainable Parameters | **616,807** |
| Training Epochs | **45** |

The CNN slightly outperformed the SVM baseline:

| Model | Test Accuracy |
|---|---:|
| SVM | 79.17% |
| **CNN** | **79.46%** |



### Per-Genre Performance

| Genre | Precision | Recall | F1 |
|---|---:|---:|---:|
| Blues | 0.76 | 0.73 | 0.75 |
| Classical | **0.93** | **0.95** | **0.94** |
| Country | 0.67 | 0.76 | 0.71 |
| Hip Hop | 0.86 | 0.84 | 0.85 |
| Jazz | 0.79 | 0.81 | 0.80 |
| Pop | 0.54 | 0.49 | 0.51 |
| Rock | 0.82 | 0.80 | 0.81 |

Classical was the strongest-performing genre, while Pop was the most difficult to classify.

## Evaluation on Unseen Music

To test how well the model generalizes beyond the GTZAN and FMA datasets, a separate collection of approximately **35–70 songs** was gathered from:

- Jamendo
- YouTube Audio Library

These songs were not included in the training, validation, or test datasets.

The new audio was processed using the same preprocessing pipeline as the training data, without augmentation.

### Unseen Data Accuracy

**77.02%**

| Genre | Precision | Recall | F1 |
|---|---:|---:|---:|
| Blues | 0.48 | 0.54 | 0.51 |
| Classical | 0.89 | 0.95 | 0.92 |
| Country | 0.71 | 0.76 | 0.74 |
| Hip Hop | 0.87 | 0.89 | 0.88 |
| Jazz | 0.70 | 0.82 | 0.75 |
| Pop | 0.68 | 0.35 | 0.46 |
| Rock | 0.80 | 0.75 | 0.77 |

The decrease from 79.46% on the original test set to 77.02% on external music suggests that the model generalizes reasonably well, while still being affected by differences in production quality, labeling conventions, and artist styles.

## Analysis

The CNN only marginally outperformed the SVM despite having over 600,000 trainable parameters.

This suggests that, with approximately 3,800 training samples, traditional hand-crafted audio features already capture many of the characteristics that distinguish the seven genres. The CNN's additional complexity may become more beneficial with a larger and more diverse dataset.

The experiments also demonstrated a tradeoff between model capacity and overfitting. Earlier versions of the model had approximately 106,000 parameters and tended toward underfitting, while the final 617,000-parameter model achieved higher training accuracy but showed a larger gap between training and validation performance.

### Genre Confusion

The confusion matrix showed that:

- **Classical** was easiest to classify, likely because of its distinctive instrumentation and structure.
- **Pop** was the most difficult genre and was frequently confused with Country, Hip Hop, and Rock.
- **Blues** was also frequently confused with Country and Jazz.

These results suggest that genres with highly distinctive spectrogram patterns are easier to separate, while genres with overlapping musical and production characteristics are inherently more difficult to distinguish.

## Technologies Used

- **Python**
- **PyTorch**
- **Librosa**
- **NumPy**
- **Convolutional Neural Networks**
- **Support Vector Machines**
- **Mel Spectrograms**
- **Audio Data Augmentation**

## Project Structure

The exact structure may vary depending on the files included in this repository, but the project follows the general workflow:

```text
Music-Genre-Categorization/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── preprocessing/
│   └── ...
│
├── models/
│   ├── cnn/
│   └── svm/
│
├── training/
│   └── ...
│
├── evaluation/
│   └── ...
│
├── README.md
└── requirements.txt
```

> **Note:** Dataset files and generated spectrograms may not be included in this repository due to dataset size and licensing considerations.

## How It Works

The overall pipeline is:

```text
Audio File
    │
    ▼
Resample to 22,050 Hz
    │
    ▼
Trim to 30 seconds
    │
    ▼
Mel Spectrogram
    │
    ▼
Normalize
    │
    ▼
CNN
    │
    ▼
Learned Audio Features
    │
    ▼
Fully Connected Layers
    │
    ▼
Genre Prediction
```

## Limitations

There are several limitations to the current system.

### Dataset Bias

GTZAN and FMA are weighted toward Western music. As a result, the model may perform poorly on genres that are underrepresented or absent from the training data, such as Afrobeats, Reggaeton, and traditional folk music.

### Genre Ambiguity

Music does not always fit cleanly into a single genre. Many songs blend multiple genres, meaning that assigning one ground-truth label can be subjective. The model consequently inherits limitations from the labels used in its training data.

### Dataset Size

The relatively small number of training samples limits the ability of a larger CNN to learn more complex relationships between similar genres.

### Copyright

Although the datasets used in development were openly available, a commercial implementation would need to ensure that all training audio is appropriately licensed.

## Future Improvements

Potential improvements to the project include:

- Training on a larger and more diverse music dataset
- Adding more genres
- Including non-Western genres
- Exploring deeper or pretrained CNN architectures
- Experimenting with transfer learning
- Improving class balancing
- Using longer or multiple audio segments per song
- Exploring multi-label genre classification instead of forcing a single genre
- Expanding the external evaluation dataset

## References

The project was informed by prior work in automated music genre classification, including the original GTZAN work by Tzanetakis and Cook and later CNN/RNN-based approaches.

Key references:

1. Tzanetakis, G. & Cook, P. (2002). *Musical Genre Classification of Audio Signals*. IEEE Transactions on Speech and Audio Processing.
2. Feng, L., Liu, S., & Yao, J. (2017). *Music Genre Classification with Paralleling Recurrent Convolutional Neural Network*.
3. Xu, W. (2024). *Music Genre Classification Using Deep Learning: A Comparative Analysis of CNNs and RNNs*.
4. Ashraf, M. et al. (2025). *Music Genre Classification with Modified Residual Learning and Dual Neural Network*.
5. Tzanetakis & Cook — GTZAN Genre Collection.
6. Free Music Archive — FMA Dataset.
7. Librosa documentation — Mel Spectrograms.

## Author

**Ireen Cheralody**  
University of Toronto  
Computer Engineering

---

### Results at a Glance

> **CNN Test Accuracy:** 79.46%  
> **SVM Baseline:** 79.17%  
> **Unseen Music Accuracy:** 77.02%  
> **Genres:** 7  
> **CNN Parameters:** 616,807

This project demonstrates an end-to-end deep learning pipeline for music genre classification, from raw audio preprocessing and spectrogram generation to CNN training, baseline comparison, and evaluation on unseen music.