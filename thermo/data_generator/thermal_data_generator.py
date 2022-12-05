import json
from pathlib import Path
from typing import Sequence, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from albumentations import Compose, Flip, Affine, KeypointParams
from scipy.ndimage import gaussian_filter


class ThermalDataset(tf.keras.utils.Sequence):
    def __init__(self, data_path: Path, sequences_names: Sequence[str], person_point_weight: float, batch_size: int, augment: bool = False):
        self._frames, self._labels = self._load_data(data_path, sequences_names)
        self._person_point_weight = person_point_weight
        self._batch_size = batch_size
        self._augment = augment
        self._augmentations = Compose([
            Affine(scale=(0.9, 1.1), rotate=(-15, 15), shear=(-5, 5), translate_percent=(-10, 10)),
            Flip()
        ], keypoint_params=KeypointParams(format='xy', remove_invisible=True))
        self._transforms = Compose([], keypoint_params=KeypointParams(format='xy', remove_invisible=True))

    @staticmethod
    def _load_data(data_path: Path, sequences_names: Sequence[str]) -> Tuple[List[np.ndarray], List[List[List[float]]]]:
        frames = []
        labels = []
        for sequence_name in sequences_names:
            df = pd.read_hdf(data_path / f'{sequence_name}.h5')

            frames.extend((np.array(df['data'].values.tolist()) - 20) / 15)
            labels.extend(df['points'].values.tolist())
        
        return frames, labels

    @staticmethod
    def generate_mask(keypoints: List[Tuple[int, int]], image_shape: Tuple[int, int], person_point_weight: float, sigma: Tuple[int, int] = (3,3)):
        label = np.zeros(image_shape, dtype=np.float32)

        for key in keypoints:
            x, y = map(int, key)
            label[y, x] = person_point_weight

        label = gaussian_filter(label, sigma=sigma, order=0)

        return np.array([label])

    def __len__(self):
        return len(self._frames) // self._batch_size

    def __getitem__(self, batch_idx: int) -> Tuple[tf.Tensor, tf.Tensor]:
        frames = []
        masks = []
        for i in range(self._batch_size):
            idx = batch_idx * self._batch_size + i

            frame, keypoints = self._frames[idx], self._labels[idx]
            for keypoint in keypoints:
                keypoint[0] = min(keypoint[0], frame.shape[1] - 1)
                keypoint[1] = min(keypoint[1], frame.shape[0] - 1)

            if self._augment:
                transformed = self._augmentations(image=frame, keypoints=keypoints)
            else:
                transformed = self._transforms(image=frame, keypoints=keypoints)

            frame, keypoints = transformed['image'], transformed['keypoints']
            mask = self.generate_mask(keypoints, frame.shape, self._person_point_weight)

            frames.append(frame)
            masks.append(mask)

        return np.expand_dims(np.stack(frames), axis=-1), np.vstack(masks)
