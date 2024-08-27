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
    def __init__(self, dataset_path: str, window_size: int):
        self.dataset = pd.read_csv(dataset_path)
        self.window_size = window_size
        self.months = []
        self.months_val = []
        self.target_indexes = [self.dataset.columns.get_loc(x) for x in TARGET_COLUMNS]

        for _, x in self.dataset.groupby('date'):
            year, month = x.iloc[0,0].split('-')
            year = year if year.startswith('20') else f'20{year}'
            self.months_val.append((year, month))
            x.drop(columns=['date'], inplace=True)
            self.months.append(x)

        assert len(self.months) - self.window_size > 0, 'Window size too large'

    def __iter__(self):
        for start_month in range(0, len(self.months) - self.window_size):
            data_train = pd.concat(self.months[start_month:start_month + self.window_size])
            data_test = self.months[start_month + self.window_size]
            self.n_features = len(data_train.columns)

            # Month to predict
            self.current_eval_year, self.current_eval_month = self.months_val[start_month + self.window_size]

            # Replace null values with the mean of the column
            data_train.fillna(data_train.mean(), inplace=True)

            data = pd.concat((data_train, data_test)).groupby(['land_mask', 'latitude', 'longitude'])

            X = []
            y = []

            for _, x in data:
                if len(x) != self.window_size + 1:
                    continue

                time_series = x.values.astype(np.float32)
                X.append(time_series[:-1])
                y.append(time_series[-1,self.target_indexes])

            X = torch.from_numpy(np.stack(X, axis=0))
            y = torch.from_numpy(np.stack(y, axis=0))

            data = PollutionDataset(X, y)
            split_len = int(len(data) * 0.8)
            train_data, test_data = random_split(data, [split_len, len(data) - split_len])
            yield train_data, test_data

    def __len__(self):
        return len(self.months) - self.window_size

    def get_month_year(self):
        return self.current_eval_month, self.current_eval_year

