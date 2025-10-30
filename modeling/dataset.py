from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from preprocessing.preprocessing_functions import (
    reindex_frames,
    handle_missing_face,
    handle_missing_hands,
    fill_missing_hands,
    drop_sequences_by_missing_face,
    resample_sequence,
    center_around_nose,
    scale_by_shoulder_width,
    scale_by_shoulder_width_2,
    center_around_nose_2
)


class SignLanguageDataset(Dataset):
    def __init__(self, df_metadata, label_mapping):
        self.df_metadata = df_metadata.reset_index(drop = True)
        self.label_mapping = label_mapping
        #self.path = Path('asl-signs')
        self.path = Path('dataset')

    def __len__(self):
        return len(self.df_metadata)

    
    def __getitem__(self, index):
        row = self.df_metadata.iloc[index]

        '''
        sequence_file_path = self.path / row['path']
        sequence_df = pd.read_parquet(sequence_file_path)
        '''

        sequence_file_path = self.path / row['path']
        features = np.load(sequence_file_path)

        '''full try
        #preprocessing on given sequence
        sequence_df = reindex_frames(sequence_df)
        sequence_df = handle_missing_face(sequence_df)
        sequence_df = handle_missing_hands(sequence_df)
        sequence_df = center_around_nose_2(sequence_df)
        sequence_df = scale_by_shoulder_width_2(sequence_df)
        sequence_df = fill_missing_hands(sequence_df)

        sequence_df = resample_sequence(sequence_df, target_frames = 60)
        '''
        '''minimal try
        sequence_df = reindex_frames(sequence_df)
        sequence_df['x'] = sequence_df['x'].fillna(0) 
        sequence_df['y'] = sequence_df['y'].fillna(0)
        sequence_df['z'] = sequence_df['z'].fillna(0)
        sequence_df = resample_sequence(sequence_df, target_frames = 30)
        '''
        '''
        #Features
        features = sequence_df[['x', 'y', 'z']].values.reshape(30, 1629) # (60,1629/-1) 543*3
        '''

        # Sign of the Sequence
        sign = row['sign']
        label = self.label_mapping[sign]


        return torch.tensor(data = features, dtype=torch.float32), torch.tensor(data = label, dtype=torch.int64)
        



