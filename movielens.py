import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

import math

from torch.utils.data import Dataset


class MovieLensDataset(Dataset):
    def __init__(self, data_path):
        self.label_encoder = LabelEncoder()
        self.data, self.labels, self.field_dims = self.load_fm_dataset(data_path)

    def load_fm_dataset(self, data_path):
        print('Reconstructing MovieLens data from {}'.format(data_path))

        header = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
        df_user = pd.read_csv(data_path + 'users.dat', sep='::', names=header, engine='python')
        for col in df_user.columns[1:]:
            df_user[col] = self.label_encoder.fit_transform(df_user[col])

        header = ['item_id', 'title', 'genres']
        df_item = pd.read_csv(data_path + 'movies.dat', sep='::',names=header, encoding="ISO-8859-1", engine='python')

        year = df_item.title.str[-5:-1].apply(lambda x: int(x))
        df_item['years'] = self.label_encoder.fit_transform(year)
        df_genres = df_item.genres.str.split('|').str.join('|').str.get_dummies()
        df_genres['genre'] = ''
        for col in df_genres.columns:
            df_genres['genre'] += df_genres[col].map(str)
            if col != 'genre':
                del df_genres[col]
        df_genres['genre'] = self.label_encoder.fit_transform(df_genres['genre'])
        df_item.drop(['title', 'genres'], axis=1, inplace=True)
        df_item = pd.concat([df_item, df_genres], axis=1)

        header = ['user_id', 'item_id', 'rating', 'timestamp']

        df_data = pd.read_csv(data_path + 'ratings.dat', sep='::', names=header, engine='python')
        df_data['timestamp'] = df_data['timestamp'] - df_data['timestamp'].min()
        df_data['timestamp'] = df_data['timestamp'].apply(self.convert_numeric_feature)
        df_data['timestamp'] = self.label_encoder.fit_transform(df_data['timestamp'])

        df_data = df_data[(~df_data['rating'].isin([3]))].reset_index(drop=True)
        df_data['rating'] = df_data.rating.apply(lambda x: 1 if int(x) > 3 else 0)

        df_data = df_data.merge(df_user, on='user_id', how='left')
        df_data = df_data.merge(df_item, on='item_id', how='left')

        data, labels = df_data.iloc[:, 3:], df_data['rating']

        field_dims = data.nunique()

        return data.values, labels.values, field_dims.values

    def convert_numeric_feature(self, val):
        if val == '':
            return 'NULL'
        v = int(val)
        if v > 2:
            return str(int(math.log(v) ** 2))
        else:
            return str(v - 2)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
