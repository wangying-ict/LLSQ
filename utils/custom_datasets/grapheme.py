"""
kaggle:
    https://www.kaggle.com/c/bengaliai-cv19/data
"""

import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pyarrow.parquet as pq
import ipdb

__all__ = ['Grapheme']


class Grapheme(Dataset):
    def __init__(self, root, transform=None, target_transform=None, _type='test'):
        if _type == 'train':
            # 25148 segmentation fault (core dumped)
            # conda install pyarrow rather than pip.
            train = pd.read_csv(os.path.join(root, 'train.csv'))
            data0 = pq.read_pandas(os.path.join(root, 'train_image_data_0.parquet')).to_pandas()
            data1 = pq.read_pandas(os.path.join(root, 'train_image_data_1.parquet')).to_pandas()
            data2 = pq.read_pandas(os.path.join(root, 'train_image_data_2.parquet')).to_pandas()
            data3 = pq.read_pandas(os.path.join(root, 'train_image_data_3.parquet')).to_pandas()
            data_full = pd.concat([data0, data1, data2, data3], ignore_index=True)
        else:
            test = pd.read_csv(os.path.join(root, 'test.csv'))
            data0 = pq.read_pandas(os.path.join(root, 'test_image_data_0.parquet')).to_pandas()
            data1 = pq.read_pandas(os.path.join(root, 'test_image_data_1.parquet')).to_pandas()
            data2 = pq.read_pandas(os.path.join(root, 'test_image_data_2.parquet')).to_pandas()
            data3 = pq.read_pandas(os.path.join(root, 'test_image_data_3.parquet')).to_pandas()
            data_full = pd.concat([data0, data1, data2, data3], ignore_index=True)

        ipdb.set_trace()
        self.df = None
        self.label = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label1 = self.label.vowel_diacritic.values[idx]
        label2 = self.label.grapheme_root.values[idx]
        label3 = self.label.consonant_diacritic.values[idx]
        image = self.df.iloc[idx][1:].values.reshape(64, 64).astype(np.float)
        return image, label1, label2, label3
