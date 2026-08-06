import torch
import numpy as np
from torch.utils.data import Dataset


class SpecDataset(Dataset):
    def __init__(self, X, y, train=False):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)
        self.train = train  # only apply SpecAugment to training data

    def __len__(self):
        return len(self.X)

    def spec_augment(self, spec, freq_mask_param=15, time_mask_param=40, n_masks=2):
        spec = spec.clone()
        n_mels, n_frames = spec.shape[1], spec.shape[2]

        for _ in range(n_masks):
            # frequency mask
            f = np.random.randint(0, freq_mask_param)
            f0 = np.random.randint(0, max(1, n_mels - f))
            spec[:, f0:f0 + f, :] = 0

            # time mask
            t = np.random.randint(0, time_mask_param)
            t0 = np.random.randint(0, max(1, n_frames - t))
            spec[:, :, t0:t0 + t] = 0

        return spec

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.train:
            x = self.spec_augment(x)
        return x, self.y[idx]