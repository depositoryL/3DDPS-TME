import math
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler


def build_dataloader(args, config):
    train_dataset = TMEDataset(args, period='train')
    test_dataset = TMEDataset(args, period='test')

    # rm, rm_pinv, a, b = train_dataset.rm, train_dataset.rm_pinv, train_dataset.scale, train_dataset.shift
    rm, rm_pinv, a, b = train_dataset.rm, train_dataset.rm_pinv, train_dataset.traffic, train_dataset.traffic
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['dataloader']['batch_size'], shuffle=True,
                              num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['dataloader']['test_size'], shuffle=False,
                             num_workers=0)

    # return train_loader, test_loader, rm, rm_pinv, torch.from_numpy(a).float(), torch.from_numpy(b).float()
    return train_loader, test_loader, rm, rm_pinv, a, b


def build_dataloader_sequence(args, config):
    train_dataset = TMESeqDataset(args, period='train')
    test_dataset = TMESeqDataset(args, period='test')

    # rm, a, b = train_dataset.rm, train_dataset.scale, train_dataset.shift
    rm, a, b = train_dataset.rm, train_dataset.traffic, train_dataset.traffic

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['dataloader']['batch_size'], shuffle=True,
                              num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['dataloader']['test_size'], shuffle=False,
                             num_workers=0)

    # return train_loader, test_loader, rm, torch.from_numpy(a).float(), torch.from_numpy(b).float()
    return train_loader, test_loader, rm, a, b


class TMEDataset(Dataset):
    def __init__(self, args, period='train'):
        super(TMEDataset, self).__init__()

        assert period in ['train', 'test'], ''
        self.period = period

        # traffic, self.link, self.rm, self.scaler = self.read_data(args)
        traffic, self.link, self.rm = self.read_data(args)
        # self.scale, self.shift = self.scaler.scale_, self.scaler.min_
        # if period == 'train':
        #     self.traffic = self.scaler.transform(traffic)
        self.traffic = torch.from_numpy(traffic).float()
        self.rm_pinv = torch.linalg.pinv(self.rm)

        self.len, self.ll_dim = self.link.shape
        _, self.tm_dim = self.traffic.shape

    def read_data(self, args):
        """ Reads a single .csv """
        df = pd.read_csv(args.tm_filepath, header=None)
        df.drop(df.columns[-1], axis=1, inplace=True)
        traffic = df.values

        if args.clip:
            traffic = self.preprocess(traffic)

        traffic = traffic[:(args.train_size + args.test_size)] / args.scale
        # scaler = MinMaxScaler()
        # scaler.fit(traffic)

        if self.period == 'test':
            traffic = traffic[args.train_size:, :]
        else:
            traffic = traffic[:args.train_size, :]

        rm_df = pd.read_csv(args.rm_filepath, header=None)
        rm_df.drop(rm_df.columns[-1], axis=1, inplace=True)
        rm = torch.from_numpy(rm_df.values).float()
        link = torch.from_numpy(traffic).float() @ rm

        # return traffic, link, rm, scaler
        return traffic, link, rm

    def preprocess(self, data):
        dim = data.shape[-1]
        for i in range(dim):
            data_i = data[:, i]
            data_i = data_i[data_i != 0]
            if data_i.shape[0] != 0:
                quantile_i = np.percentile(data_i, 99.5)
                data[data[:, i] > quantile_i, i] = quantile_i
        return data

    def __getitem__(self, ind):
        x = self.traffic[ind, :]
        if self.period == 'train':
            return x
        y = self.link[ind, :]
        return x, y

    def __len__(self):
        return self.len


class TMESeqDataset(Dataset):
    def __init__(
            self,
            args,
            period='train'
    ):
        super(TMESeqDataset, self).__init__()
        assert period in ['train', 'test_train', 'test'], ''
        self.period, self.window = period, args.window

        # traffic, self.link, self.rm, self.scaler = self.read_data(args)
        traffic, self.link, self.rm = self.read_data(args)
        # self.scale, self.shift = self.scaler.scale_, self.scaler.min_
        # if period == 'train':
        #     self.traffic = self.scaler.transform(traffic)
        self.traffic = torch.from_numpy(traffic).float()

        self.len, self.ll_dim = self.link.shape
        _, self.tm_dim = self.traffic.shape
        size_sqrt = int(math.sqrt(self.tm_dim))

        if period == 'train':
            self.sample_num = max(self.len - self.window + 1, 0)
        else:
            assert self.len % self.window == 0, 'Filed to build test dataset owing to the length of time-series.'
            self.sample_num = int(self.len / self.window) if int(self.len / self.window) > 0 else 0

        self.traffic, self.link = self.__getsamples(self.traffic, self.link, period)
        if period == 'train':
            self.traffic = self.traffic.reshape(-1, 1, self.window, size_sqrt, size_sqrt)

    def read_data(self, args):
        """ Reads a single .csv """
        df = pd.read_csv(args.tm_filepath, header=None)
        df.drop(df.columns[-1], axis=1, inplace=True)
        traffic = df.values

        if args.clip:
            traffic = self.preprocess(traffic)

        traffic = traffic[:(args.train_size + args.test_size)] / args.scale
        # scaler = MinMaxScaler()
        # scaler.fit(traffic)

        if self.period == 'test':
            traffic = traffic[args.train_size:, :]
        else:
            traffic = traffic[:args.train_size, :]

        rm_df = pd.read_csv(args.rm_filepath, header=None)
        rm_df.drop(rm_df.columns[-1], axis=1, inplace=True)
        rm = torch.from_numpy(rm_df.values).float()
        link = torch.from_numpy(traffic).float() @ rm

        # return traffic, link, rm, scaler
        return traffic, link, rm

    def preprocess(self, data):
        dim = data.shape[-1]
        for i in range(dim):
            data_i = data[:, i]
            data_i = data_i[data_i != 0]
            if data_i.shape[0] != 0:
                quantile_i = np.percentile(data_i, 99.5)
                data[data[:, i] > quantile_i, i] = quantile_i
        return data

    def __getitem__(self, ind):
        x = self.traffic[ind, :, :]
        if self.period == 'train':
            return x
        y = self.link[ind, :, :]
        return x, y

    def __len__(self):
        return self.sample_num

    def __getsamples(self, data1, data2, period):
        x1 = torch.zeros((self.sample_num, self.window, self.tm_dim))
        x2 = torch.zeros((self.sample_num, self.window, self.ll_dim))
        if period == 'train':
            for i in range(self.sample_num):
                start = i
                end = i + self.window
                x1[i, :, :] = data1[start:end, :]
                x2[i, :, :] = data2[start:end, :]
        else:
            j = 0
            for i in range(0, self.sample_num):
                start = j
                end = j + self.window
                x1[i, :, :] = data1[start:end, :]
                x2[i, :, :] = data2[start:end, :]
                j = end

        return x1, x2
