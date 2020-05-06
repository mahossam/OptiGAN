import numpy as np
import random


class RealDataLoader():
    def __init__(self, batch_size, seq_length, end_token=0):
        self.batch_size = batch_size
        self.token_stream = []
        self.seq_length = seq_length
        self.end_token = end_token

    def create_batches(self, data_file, data_file_no_padding, count=-1):
        self.token_stream = []
        self.token_stream_no_padding = list()

        with open(data_file, 'r') as raw:
            d_index = 0
            for line in raw:
                if count > 0 and d_index > count:
                    break
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                if len(parse_line) > self.seq_length:
                    self.token_stream.append(parse_line[:self.seq_length])
                else:
                    while len(parse_line) < self.seq_length:
                        parse_line.append(self.end_token)
                    if len(parse_line) == self.seq_length:
                        self.token_stream.append(parse_line)
                d_index = d_index + 1

        with open(data_file_no_padding, 'r') as raw:
            d_index = 0
            for line in raw:
                if count > 0 and d_index > count:
                    break
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                self.token_stream_no_padding.append(parse_line[:self.seq_length])
                d_index = d_index + 1

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.token_stream_no_padding = self.token_stream_no_padding[:self.num_batch * self.batch_size]
        self.sequence_batches = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.np_seq_batches = np.array(self.sequence_batches)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def random_batch(self):
        rn_pointer = random.randint(0, self.num_batch - 1)
        ret = self.sequence_batches[rn_pointer]
        return ret

    def random_some(self, n, max_length):
        # rn_pointer = random.randint(0, self.num_batch - 1)
        rand_list = list()
        n_batches = int(n/self.batch_size)
        if n_batches > self.num_batch:
            n_batches = self.num_batch
        filters = np.random.choice(self.num_batch, int(n/self.batch_size), replace=False)
        return self.flatten_loa(self.np_seq_batches[filters, :,:max_length])

    def get_as_lol_no_padding(self):
        return self.token_stream_no_padding

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