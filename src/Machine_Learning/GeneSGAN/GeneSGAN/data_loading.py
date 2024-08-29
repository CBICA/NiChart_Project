import torch
import numpy as np

__author__ = "Zhijian Yang"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Zhijian Yang"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Zhijian Yang"
__email__ = "zhijianyang@outlook.com"
__status__ = "Development"

class Paired_data_iterator(object):
    def __init__(self, Gene_data, Gene_data_fillna, PT_data, fraction, batch_size, random_seed):
        super(Paired_data_iterator, self).__init__()
        self.pt_data = torch.from_numpy(PT_data.astype('float32'))
        self.gene_data = torch.from_numpy(Gene_data.astype('float32'))
        self.gene_data_fillna = torch.from_numpy(Gene_data_fillna.astype('float32'))
        self.num_samples = self.pt_data.shape[0]
        self.batch_size = batch_size
        self.n_batches = int(self.num_samples / self.batch_size)
        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        self.data_indices = np.random.permutation(self.num_samples)
        self.batch_idx = 0

    def next(self):
        if self.batch_idx == self.n_batches-1:
            self.reset()
            #raise StopIteration

        idx = self.batch_idx * self.batch_size
        chosen_indices = self.data_indices[idx:idx+self.batch_size]

        self.batch_idx += 1

        return {'gene': self.gene_data[chosen_indices],'gene_filled':self.gene_data_fillna[chosen_indices],'y': self.pt_data[chosen_indices]}

    def __len__(self):
        return self.num_samples

class PTIterator(object):
    def __init__(self, PT_data, batch_size):
        super(PTIterator, self).__init__()
        self.data = torch.from_numpy(PT_data.astype('float32'))
        self.num_samples = self.data.shape[0]
        self.batch_size = batch_size
        self.n_batches = int(self.num_samples / self.batch_size)
        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        self.data_indices = np.random.permutation(self.num_samples)
        self.batch_idx = 0

    def __next__(self):
        if self.batch_idx == self.n_batches-1:
            self.reset()
            raise StopIteration

        idx = self.batch_idx * self.batch_size
        chosen_indices = self.data_indices[idx:idx+self.batch_size]

        self.batch_idx += 1

        return {'y': self.data[chosen_indices]}

    def __len__(self):
        return self.num_samples

class CNIterator(object):
    def __init__(self, CN_data, batch_size):
        super(CNIterator, self).__init__()
        self.data = torch.from_numpy(CN_data.astype('float32'))
        self.num_samples = self.data.shape[0]
        self.batch_size = batch_size
        self.n_batches = int(self.num_samples / self.batch_size)
        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        self.data_indices = np.random.permutation(self.num_samples)
        self.batch_idx = 0

    def next(self):
        if self.batch_idx == self.n_batches:
            self.reset()
            #raise StopIteration

        idx = self.batch_idx * self.batch_size
        chosen_indices = self.data_indices[idx:idx+self.batch_size]

        self.batch_idx += 1

        return {'x': self.data[chosen_indices]}

    def __len__(self):
        return self.num_samples

class val_PT_construction(object):
    def __init__(self, PT_data, random_seed, fraction, **kwargs):
        super(val_PT_construction, self).__init__()
        np.random.seed(random_seed)
        indices = np.random.choice(PT_data.shape[0], int(fraction*PT_data.shape[0]), replace=False)
        select = np.in1d(range(PT_data.shape[0]), indices)
        self.data = torch.from_numpy(PT_data.astype('float32'))
        self.train_index = select

    def load_train(self):
        return self.data[self.train_index]

    def load_test(self):
        return self.data[~self.train_index]

    def __len__(self):
        return self.num_samples


class val_CN_construction(object):
    def __init__(self, CN_data, random_seed, fraction,**kwargs):
        super(val_CN_construction, self).__init__()
        np.random.seed(random_seed)
        indices = np.random.choice(CN_data.shape[0], int(fraction*CN_data.shape[0]), replace=False)
        select = np.in1d(range(CN_data.shape[0]), indices)
        self.data = torch.from_numpy(CN_data.astype('float32'))
        self.train_index = select

    def load_train(self):
        return self.data[self.train_index]

    def load_test(self):
        return self.data[~self.train_index]

    def __len__(self):
        return self.num_samples

class val_Gene_construction(object):
    def __init__(self, Gene_data, random_seed, fraction,**kwargs):
        super(val_Gene_construction, self).__init__()
        np.random.seed(random_seed)
        indices = np.random.choice(Gene_data.shape[0], int(fraction*Gene_data.shape[0]), replace=False)
        select = np.in1d(range(Gene_data.shape[0]), indices)
        self.data = torch.from_numpy(Gene_data.astype('float32'))
        self.train_index = select

    def load_train(self):
        return self.data[self.train_index]

    def load_test(self):
        return self.data[~self.train_index]

    def __len__(self):
        return self.num_samples

