import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split

TARGET_COLUMNS = ['O3', 'PM10', 'PM2.5', 'CO', 'SO2', 'NO']

class PollutionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SlidingWindow:
    def __init__(self, dataset_path: str, window_size: int, generator=None):
        self.dataset = pd.read_csv(dataset_path)
        self.window_size = window_size
        self.generator = generator
        self.months = [x for _, x in self.dataset.groupby('date')]

        assert len(self.months) - self.window_size >= 0, 'Window size too large'

    def __iter__(self):
        for start_month in range(0, len(self.months) - self.window_size):
            data_train = pd.concat(self.months[start_month:start_month + self.window_size])
            data_test = self.months[start_month + self.window_size]
            features = [x for x in self.dataset.columns if x != 'date']
            self.n_features = len(features)

            # Month to predict
            self.current_eval_year, self.current_eval_month = data_test.iloc[0,0].split('-')
            self.current_eval_year = self.current_eval_year if self.current_eval_year.startswith('20') else f'20{self.current_eval_year}'

            data = pd.concat((data_train, data_test)).groupby(['land_mask', 'latitude', 'longitude'])

            X = torch.empty((len(data), self.window_size, self.n_features))
            y = torch.empty((len(data), len(TARGET_COLUMNS)))

            i = 0
            for _, x in data:
                if len(x) != self.window_size + 1:
                    continue
                X[i,:,:] = torch.from_numpy(x.loc[x.index[:-1],features].astype(np.float32).values)
                y[i,:] = torch.from_numpy(x.loc[x.index[-1],TARGET_COLUMNS].astype(np.float32).values)
                i += 1

            data = PollutionDataset(X[:i], y[:i])
            train_data, test_data = random_split(data, [int(i * 0.8), i - int(i * 0.8)], generator=self.generator)
            yield train_data, test_data

    def __len__(self):
        return len(self.months) - self.window_size

    def get_month_year(self):
        return self.current_eval_month, self.current_eval_year

