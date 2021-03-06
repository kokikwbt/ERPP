import datetime
from datetime import timedelta
import pandas as pd
from tqdm import tqdm, trange
import numpy as np
import torch
from sklearn import preprocessing
from collections import Counter
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softmax(x):
    x = np.exp(x)
    return x / x.sum()

def embdim(x):
    return int(2 * np.log(x) + 1)


class EventHandler:
    def __init__(self, data, config):
        """
            data: a DataFrame object with columns:
                date,event,ctg1,...,ctgk,num1,...,numk
            config: a named tuple given by main.py
        """

        self.config = vars(config)

        # Preprocessing
        data = self.encode_timestamp(data, config.time_col)
        data = self.encode_label(data, [config.event_col])

        self.data = data  # DataFrame
        self.time = data[config.time_col]
        self.event = data[config.event_col]
        self.ev_seq_len = config.ev_seq_len
        self.ts_seq_len = config.ts_seq_len

        # Preparing for input features
        self.set_features(data, config)
        self.set_feature_stats(data, self.config)
        self.generate_sequence()

        self.statistic()  # Visualize

    def set_features(self, data, config):

        self.ev_ctg_features = []
        self.ev_num_features = []
        self.ts_ctg_features = []
        self.ts_num_features = []

        if config.ev_ctg_features is not None:
            self.ev_ctg_features = config.ev_ctg_features.split('/')
            self.ev_ctg_data, self.ev_ctg_encoder = self.set_label_encoder(
                data[self.ev_ctg_features], self.ev_ctg_features)


        if config.ev_num_features is not None:
            self.ev_num_features = config.ev_num_features.split('/')
            self.ev_num_data = self.data[self.ev_num_features]

        if config.ts_ctg_features is not None:
            self.ts_ctg_features = config.ts_ctg_features.split('/')
            timeseries = data.set_index(config.time_col)
            timeseries = timeseries[self.ts_ctg_features]
            timeseries = timeseries.resample(config.freq).first()
            timeseries = timeseries.fillna('UNKNOWN')
            self.ctg_timeseries, self.ts_ctg_encoder = self.set_label_encoder(
                timeseries, self.ts_ctg_features)

        if config.ts_num_features is not None:
            self.ts_num_features = config.ts_num_features.split('/')
            timeseries = data.set_index(config.time_col)
            timeseries = timeseries[self.ts_num_features]
            timeseries = timeseries.resample(config.freq).sum()
            timeseries = timeseries.fillna(0)
            self.num_timeseries = timeseries

    def set_feature_stats(self, data, config):

        config['event_class'] = data[config['event_col']].nunique()

        ev_embdim_total = 0
        for col in self.ev_ctg_features:
            config['ev_ndim_{}'.format(col)] = data[col].nunique()
            config['ev_embdim_{}'.format(col)] = embdim(
                config['ev_ndim_{}'.format(col)])
            ev_embdim_total += config['ev_embdim_{}'.format(col)]

        config['ev_embdim_total'] = ev_embdim_total

        ts_embdim_total = 0
        for col in self.ts_ctg_features:
            config['ts_ndim_{}'.format(col)] = self.ctg_timeseries[col].nunique()
            config['ts_embdim_{}'.format(col)] = embdim(
                config['ts_ndim_{}'.format(col)])
            ts_embdim_total += config['ts_embdim_{}'.format(col)]

        config['ts_embdim_total'] = ts_embdim_total

        if self.ev_num_features:
            config['ev_ndim_num'] = len(self.ev_num_features)

        if self.ts_num_features:
            config['ts_ndim_num'] = len(self.ts_num_features)

        config['ev_ctg_features'] = self.ev_ctg_features
        config['ev_num_features'] = self.ev_num_features
        config['ts_ctg_features'] = self.ts_ctg_features
        config['ts_num_features'] = self.ts_num_features

        print(config)

    def encode_timestamp(self, data, col):
        data[col] = pd.to_datetime(data[col])
        return data

    def encode_label(self, data, columns):
        for col in columns:
            le = preprocessing.LabelEncoder()
            data[col] = le.fit_transform(data[col])
            del le

        return data

    def set_label_encoder(self, data, columns):
        le_dict = {}

        for col in columns:
            le = preprocessing.LabelEncoder()
            data[col] = le.fit_transform(data[col])
            le_dict[col] = le

        return data, le_dict

    # TODO: list of dataframes
    def generate_sequence(self):

        time_seqs = []  # time points
        event_seqs = []
        ev_ctg_seqs = []
        ev_num_seqs = []
        ts_ctg_seqs = []
        ts_num_seqs = []

        esl = self.ev_seq_len
        data = self.data
        n = len(self.data)
        offset = self.ev_seq_len

        if self.config['freq'] == 'D':
            dt = timedelta(days=1)

        for i in trange(offset, n - 2):

            te = self.time[i] - dt  # exclude the time an event occurs
            ts = self.time[i] - dt * (self.ts_seq_len + 1)

            if self.ts_ctg_features:
                ts_ctg_seq = self.ctg_timeseries[ts:te].values
                if len(ts_ctg_seq) < self.ts_seq_len:
                    continue
                ts_ctg_seqs.append(ts_ctg_seq)

            if self.ts_num_features:
                ts_num_seq = self.num_timeseries[ts:te].values
                if len(ts_num_seq) < self.ts_seq_len:
                    continue
                ts_num_seqs.append(ts_num_seq)

            if self.ev_ctg_features:
                ev_ctg_seqs.append(
                    self.ev_ctg_data.iloc[i - esl:i].values)

            if self.ev_num_features:
                ev_num_seqs.append(
                    self.ev_num_data.iloc[i - esl:i].values)

            time_seqs.append(
                self.time[i - esl:i+2].map(lambda x: x.timestamp()).values)
            event_seqs.append(
                self.event[i - esl:i+2].values)

        print(len(time_seqs))
        print(len(ev_num_seqs))
        print(len(ev_ctg_seqs))
        print(len(ts_num_seqs))
        print(len(ts_ctg_seqs))

        self.time_seqs = time_seqs
        self.event_seqs = event_seqs
        self.ev_num_seqs = ev_num_seqs
        self.ev_ctg_seqs = ev_ctg_seqs
        self.ts_num_seqs = ts_num_seqs
        self.ts_ctg_seqs = ts_ctg_seqs

    def __getitem__(self, item):
        data = {}
        data['time'] = self.time_seqs[item]
        data['event'] = self.event_seqs[item]
        if self.config['ev_ctg_features']:
            data['ev_ctg_data'] = self.ev_ctg_seqs[item]
        if self.config['ev_num_features']:
            data['ev_num_data'] = self.ev_num_seqs[item]
        if self.config['ts_ctg_features']:
            data['ts_ctg_data'] = self.ts_ctg_seqs[item]
        if self.config['ts_num_features']:
            data['ts_num_data'] = self.ts_num_seqs[item]

        return data

    def __len__(self):
        return len(self.time_seqs)

    @staticmethod
    def to_features(batch):

        times = []
        events = []
        ctg_events = []
        num_events = []
        ctg_timeseries = []
        num_timeseries = []

        for data in batch:
            time = np.array([data['time'][0]] + data['time'])
            time = np.diff(time)
            times.append(time)

            events.append(data['event'])

            if 'ev_ctg_data' in data:
                ctg_events.append(data['ev_ctg_data'])
            if 'ev_num_data' in data:
                num_events.append(data['ev_num_data'])
            if 'ts_ctg_data' in data:
                ctg_timeseries.append(data['ts_ctg_data'])
            if 'ts_num_data' in data:
                num_timeseries.append(data['ts_num_data'])

        return (
            torch.FloatTensor(times),
            torch.LongTensor(events),
            torch.LongTensor(ctg_events),
            torch.FloatTensor(num_events),
            torch.LongTensor(ctg_timeseries),
            torch.FloatTensor(num_timeseries),
        )

    @staticmethod
    def to_timeseries_features(self, keys, batch, freq):
        return   # torch.

    def statistic(self):
        print("TOTAL SEQs:", len(self.time_seqs))
        intervals = np.diff(np.array(self.time)).astype(float) * 1e-12
        for thr in [0.001, 0.01, 0.1, 1, 10, 100]:
            print(f"< {thr} \t= {np.mean(intervals < thr)}")

    def importance_weight(self):
        count = Counter(self.event)
        percentage = [count[k] / len(self.event) for k in sorted(count.keys())]
        for i, p in enumerate(percentage):
            print(f"event{i} = {p * 100}%")
        weight = [len(self.event) / count[k] for k in sorted(count.keys())]
        return weight


class ATMDataset:
    def __init__(self, config, subset):
        """
        """
        data = pd.read_csv(f"dat/{subset}_day.csv")
        self.subset = subset  # train/test
        self.id = list(data['id'])
        self.time = list(data['time'])
        self.event = list(data['event'])
        self.config = config
        self.seq_len = config.seq_len
        self.time_seqs, self.event_seqs = self.generate_sequence()
        self.statistic()

    def generate_sequence(self):
        MAX_INTERVAL_VARIANCE = 1
        pbar = tqdm(total=len(self.id) - self.seq_len + 1)
        time_seqs = []
        event_seqs = []
        cur_end = self.seq_len - 1
        while cur_end < len(self.id):
            pbar.update(1)
            cur_start = cur_end - self.seq_len + 1
            if self.id[cur_start] != self.id[cur_end]:
                cur_end += 1
                continue

            subseq = self.time[cur_start:cur_end + 1]
            # if max(subseq) - min(subseq) > MAX_INTERVAL_VARIANCE:
            #     if self.subset == "train":
            #         cur_end += 1
            #         continue
            
            time_seqs.append(self.time[cur_start:cur_end + 1])
            event_seqs.append(self.event[cur_start:cur_end + 1])
            cur_end += 1
        return time_seqs, event_seqs

    def __getitem__(self, item):
        return self.time_seqs[item], self.event_seqs[item]

    def __len__(self):
        return len(self.time_seqs)

    @staticmethod
    def to_features(batch):
        times, events = [], []
        for time, event in batch:
            time = np.array([time[0]] + time)
            time = np.diff(time)
            times.append(time)
            events.append(event)
        return torch.FloatTensor(times), torch.LongTensor(events)

    def statistic(self):
        print("TOTAL SEQs:", len(self.time_seqs))
        # for i in range(10):
        #     print(self.time_seqs[i], "\n", self.event_seqs[i])
        intervals = np.diff(np.array(self.time))
        for thr in [0.001, 0.01, 0.1, 1, 10, 100]:
            print(f"<{thr} = {np.mean(intervals < thr)}")

    def importance_weight(self):
        count = Counter(self.event)
        percentage = [count[k] / len(self.event) for k in sorted(count.keys())]
        for i, p in enumerate(percentage):
            print(f"event{i} = {p * 100}%")
        weight = [len(self.event) / count[k] for k in sorted(count.keys())]
        return weight


def abs_error(pred, gold):
    return np.mean(np.abs(pred - gold))


def clf_metric(pred, gold, n_class):
    gold_count = Counter(gold)
    pred_count = Counter(pred)
    prec = recall = 0
    pcnt = rcnt = 0
    for i in range(n_class):
        match_count = np.logical_and(pred == gold, pred == i).sum()
        if gold_count[i] != 0:
            prec += match_count / gold_count[i]
            pcnt += 1
        if pred_count[i] != 0:
            recall += match_count / pred_count[i]
            rcnt += 1
    prec /= pcnt
    recall /= rcnt
    print(f"pcnt={pcnt}, rcnt={rcnt}")
    f1 = 2 * prec * recall / (prec + recall)
    return prec, recall, f1
