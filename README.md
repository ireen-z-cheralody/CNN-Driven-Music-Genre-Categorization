# Music Genre Categorization with CNN

A deep learning project that classifies music into seven genres using **Convolutional Neural Networks (CNNs)** trained on mel spectrograms generated from audio recordings.

## Overview

This project uses deep learning to automatically predict the genre of a song from its audio. Each 30-second audio clip is converted into a normalized mel spectrogram, which is then treated as an image and classified using a CNN.
The project also implements a **Support Vector Machine (SVM)** as a traditional machine learning baseline and evaluates the CNN on completely unseen music.

## Results

| Model | Accuracy |
|---|---:|
| SVM Baseline | 79.17% |
| **CNN** | **79.46%** |
| CNN on Unseen Music | **77.02%** |

The final CNN achieved a **79.46% test accuracy** across seven genres and generalized to **77.02% accuracy on external, unseen music**.

## Genres

The model classifies music into:

- Blues
- Classical
- Country
- Hip Hop
- Jazz
- Pop
- Rock

## How It Works

```text
  Audio File
       ↓
Resample to 22,050 Hz
       ↓
30-Second Audio Clip
       ↓
Mel Spectrogram
       ↓
Normalization
       ↓
      CNN
       ↓
Genre Prediction
```

Audio was processed using `librosa` and converted into **128-band mel spectrograms**. Training data was augmented using time stretching, pitch shifting, and additive Gaussian noise.

## Model

The final CNN consists of four convolutional blocks with increasing channel depth:

```text
Input: 128 × 640 Mel Spectrogram
           ↓
Convolutional Block — 32 channels
           ↓
Convolutional Block — 64 channels
           ↓
Convolutional Block — 128 channels
           ↓
Convolutional Block — 256 channels
           ↓
Global Average Pooling
           ↓
Fully Connected Layer — 64 units
           ↓
Output Layer — 7 Genres
```

Each convolutional block uses convolution, batch normalization, ReLU activation, max pooling, and dropout. Global Average Pooling is used instead of flattening to reduce the number of parameters. The final model contains approximately **616,807 trainable parameters**.

## Dataset

The project combines:

- **GTZAN Genre Collection** (1,000 tracks)
- **Free Music Archive (FMA) Medium** (approximately 25,000 tracks)

Only the seven genres shared between the datasets were used. After preprocessing and augmentation, the dataset contained **3,808 training samples, 408 validation samples, and 409 test samples**.

## Technologies

- Python
- PyTorch
- Librosa
- NumPy
- CNN
- SVM
- Mel Spectrograms

## Project Structure

```text
Music-Genre-Categorization/
│
├── data/
├── models/
├── preprocessing/
├── training/
├── evaluation/
├── README.md
└── requirements.txt
```

> Dataset files are not included in this repository due to their size and licensing considerations.

## Key Findings

The CNN only slightly outperformed the SVM baseline (**79.46% vs. 79.17%**). This suggests that, with the relatively small training dataset, traditional audio features already capture many of the characteristics needed to distinguish the seven genres.
Classical was the strongest-performing genre, while Pop was the most difficult. Similar genres such as Pop, Blues, and Country were frequently confused with one another.

## Limitations & Future Improvements

- Expand the dataset with more diverse music and genres
- Include more non-Western genres
- Experiment with larger or pretrained CNN architectures
- Explore transfer learning
- Investigate multi-label genre classification
- Expand the external unseen-data evaluation set

## Author

**Ireen Cheralody**  
University of Toronto — Computer Engineering
