# from config import *

# from gensim.models import KeyedVectors

# from nltk.corpus import stopwords
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from datasets import generate_x_y_data_v1

import numpy as np
import glob


class Trajc_DataLoader:
    def __init__(self, batch_size, seq_length):
        # self.data_root_dir = data_root_dir
        # self.dataset_name = dataset_name
        # self.check_data_exist(dataset_name)
        self.batch_size = batch_size
        self.token_stream = list()
        self.seq_length = seq_length

    def dataset_prefixed_string(self, dataset_name, s):
        return dataset_name+"_"+s

    def _normalize_and_scale(self, train_df):
        # remove string columns and convert to float
        train_df = train_df.astype(float, copy=False)
        # MinMaxScaler - center and scale the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_df_scale = scaler.fit_transform(train_df)
        return (scaler, train_df_scale)

    def prepare_DST(self, data_root_dir, fighter_to_load="both"):
        cols = ["x", "y", "z", "psi", "theta", "phi", "v", "gload"]
        raw_time_steps = 46
        time_steps = 40
        print(data_root_dir)
        allFiles = glob.glob(data_root_dir + "/*.csv")
        data_list = list()
        n_samples = len(allFiles)
        print(n_samples)
        # start_token_value = 0.0
        # start_token = np.array([start_token_value] * len(cols))

        for file_ in allFiles:
            df = pd.read_csv(file_, header=0)
            viper_df = df[df.callsign == 'viper'].set_index('timestep')
            cobra_df = df[df.callsign == 'cobra'].set_index('timestep')
            viper_df = viper_df.head(time_steps) # take only first 40 time steps, discard rest
            cobra_df = cobra_df.head(time_steps)

            # viper_df = pd.DataFrame(np.insert(viper_df[cols].values, 0, start_token, 0), columns=cols)
            # cobra_df = pd.DataFrame(np.insert(cobra_df[cols].values, 0, start_token, 0), columns=cols)

            both_df = pd.concat([viper_df[cols], cobra_df[cols]], axis=1, ignore_index=True)
            data_list.append(both_df)

        frame = pd.concat(data_list, axis=0, ignore_index=True)
        # mydata = np.reshape(frame.values,(-1, time_steps+1, len(cols)*2))
        mydata = np.reshape(frame.values,(-1, time_steps, len(cols)*2))

        mydata_scaled = None
        # if Config.args.normz == 'featr_and_time':
        #     # scaler_t = MinMaxScaler(feature_range=(0, 1.0))
        #     # mydata_scaled = scaler_t.fit_transform(mydata[:, 0, :])
        #     # # mydata_scaled = mydata[:, 0, :]
        #     # scalers = list()
        #     # scalers.append(scaler_t)
        #     # # for i in range(1, time_steps+1):
        #     # for i in range(1, time_steps):
        #     #     scaled_t = scaler_t.fit_transform(mydata[:, i, :])
        #     #     # scaled_t = mydata[:, i, :]
        #     #     mydata_scaled = np.concatenate((mydata_scaled, scaled_t), 1)
        #     #     scalers.append(scaler_t)
        #     print("Wrong normalization. Not implemented")
        #     exit(0)

        # if Config.args.normz == 'featr_only':
        mydata = np.reshape(mydata, [-1, len(cols)*2])
        scaler_t = MinMaxScaler(feature_range=(0, 1.0))
        mydata_scaled = scaler_t.fit_transform(mydata)
        scalers = list()
        scalers.append(scaler_t)

        # elif Config.args.normz == 'featr_persample':
        #     scalers = list()
        #     mydata_scaled = np.zeros_like(mydata)
        #     for s in range(mydata.shape[0]):
        #         sample = mydata[s, :, :]
        #         # sample = np.reshape(sample, [-1, len(cols) * 2])
        #         scaler_t = MinMaxScaler(feature_range=(0, 1.0))
        #         sample_scaled = scaler_t.fit_transform(sample)
        #         scalers.append(scaler_t)
        #         mydata_scaled[s, :, :] = sample_scaled

                # mydata_final = np.reshape(mydata_scaled, (-1, time_steps+1, len(cols)*2))
        mydata_final = np.reshape(mydata_scaled, (-1, time_steps, len(cols)*2))

        # scaler, norm_data = self._normalize_and_scale(frame)
        # resh_norm_data = np.reshape(norm_data,[-1, time_steps, len(cols)])
        self.scalers = scalers
        return n_samples, mydata_final[-int(n_samples*0.20):], int(n_samples*0.20), mydata_final[:int(n_samples*0.80)], int(n_samples*0.80), len(cols)*2

    def create_batches(self, data_root_dir, dataset_name):
        # data_root_dir = sys.argv[1]
        # dataset_name = sys.argv[3]

        # self.check_data_exist(dataset_name)

        data_array = None

        data_size, test, test_size, train, train_size, input_size = self.prepare_DST(data_root_dir)

        self.train = train
        self.test = test
        self.input_size = input_size
        self.data_size = data_size

        self.num_batch = int(train_size / self.batch_size)
        # self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        # self.token_stream_no_padding = self.token_stream_no_padding[:self.num_batch * self.batch_size]
        self.sequence_batches = np.split(np.array(self.train), self.num_batch, 0)
        self.np_seq_batches = np.array(self.sequence_batches)
        self.pointer = 0

        # return train, test, data_size, train_size, test_size, input_size

    # def create_batches(self, data_file, data_file_no_padding, count=-1):
    #     self.token_stream = []
    #     self.token_stream_no_padding = list()
    #
    #     with open(data_file, 'r') as raw:
    #         d_index = 0
    #         for line in raw:
    #             if count > 0 and d_index > count:
    #                 break
    #             line = line.strip().split()
    #             parse_line = [int(x) for x in line]
    #             if len(parse_line) > self.seq_length:
    #                 self.token_stream.append(parse_line[:self.seq_length])
    #             else:
    #                 while len(parse_line) < self.seq_length:
    #                     parse_line.append(self.end_token)
    #                 if len(parse_line) == self.seq_length:
    #                     self.token_stream.append(parse_line)
    #             d_index = d_index + 1
    #
    #     with open(data_file_no_padding, 'r') as raw:
    #         d_index = 0
    #         for line in raw:
    #             if count > 0 and d_index > count:
    #                 break
    #             line = line.strip().split()
    #             parse_line = [int(x) for x in line]
    #             self.token_stream_no_padding.append(parse_line[:self.seq_length])
    #             d_index = d_index + 1
    #
    #     self.num_batch = int(len(self.token_stream) / self.batch_size)
    #     self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
    #     self.token_stream_no_padding = self.token_stream_no_padding[:self.num_batch * self.batch_size]
    #     self.sequence_batches = np.split(np.array(self.token_stream), self.num_batch, 0)
    #     self.np_seq_batches = np.array(self.sequence_batches)
    #     self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def random_batch(self):
        rn_pointer = random.randint(0, self.num_batch - 1)
        ret = self.sequence_batches[rn_pointer]
        return ret

    def reset_pointer(self):
        self.pointer = 0

    def flatten_lol(self, lol):
        flat_l = []
        for l in lol:
            flat_l += l
        return flat_l

    def flatten_loa(self, loa):
        flat_l = list()
        for a in loa:
            for inner_a in a:
                flat_l.append(inner_a)
        return flat_l
