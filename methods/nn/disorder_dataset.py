import os
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import numpy as np

from ..utils.read_embeddings import read_data
from torch.nn.utils.rnn import pad_sequence

# TO DO: setup.py

project_root = os.getcwd()

class DisorderDataset(Dataset):

    def __init__(self, x, y):

        """ 
        In: 
            x: list containing embeddings of size # of proteins
            y: array containing z-scores

        The order of proteins is identical in both
        """

        self.x = x
        self.y = y
        self.avg_y = self.calculate_avg_z(y)
        self.aa_len = np.array([len(seq) for seq in x])
        self.bins = self.assign_bins(self.avg_y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):

        """
        In:
            i: an integer value to index data
        Outs:
            data: A dictionary of {x,y}
            """
        embeddings = self.x[i]
        z_scores = self.y[i]

        return {
            'embeddings': torch.tensor(embeddings).float(),
            'z_scores': torch.tensor(z_scores).float(),
        }

    
    '''
    Calculates an average z_score per aa sequence. Unknown z-scores are
    ignored.

    Args:
        In:
            y: list, list of lists of length #of aa sequences
                entries are lists which contain z-score for each aa of a sequence.
        Out:
            avg_zs: list, list of averaged z-scores per sequence 
    '''

    def calculate_avg_z(self,y):

        unknown_z_value = 999.0
        avg_zs = []

        for z_scores_per_seq in y:
            # seq length after removing unknown z-scores 
            trimmed_length = 0
            # sum of all defined z-scores
            total_z_sum = 0

            # filter out the 999.0 unnknown z-scores 
            for z_score in z_scores_per_seq:
                if z_score != 999.0:
                    trimmed_length += 1
                    total_z_sum += z_score

            avg_zs.append(total_z_sum/trimmed_length)

        return np.array(avg_zs)

    ''' Assign bins (int class indicator) to each of samples according
    to histogram of average z-scores.
        Args:
        In:
            avg_z: list of average z-scores per sample
            bins_num: number of bins in a histogram
        Out:
            assigned_bins: list of integers indicating an artificial class label
         '''

    def assign_bins(self,avg_z, bins_num=4):
        samples_per_bin, bins, = np.histogram(avg_z, bins=4)
        print("Samples per bin", samples_per_bin)
        print("Thresholds of bins", bins)
        assigned_bins = []
        # TO DO: bins_num is dummy: implement for any number of bins
        for z_score in avg_z:
            if z_score < bins[1]:
                assigned_bins.append(1)
            elif z_score < bins[2]:
                assigned_bins.append(2)
            elif z_score < bins[3]:
                assigned_bins.append(3)
            elif z_score <= bins[4]:
                assigned_bins.append(4)
            else:
                print("Error! Z-score out of boundaries.")
                break
        return np.array(assigned_bins)


def load_dataset(path=os.path.join(project_root, "data"), window_size=7):

    z_score_path = 'baseline_embeddings_disorder.h5'

    labels_path = 'disorder_labels.fasta'

    x, y = read_data(os.path.join(path, z_score_path),
                    os.path.join(path, labels_path))

    # adding a padding of size (window_size-1)/2 so that we can have a sliding window which also incorporates
    # residues at the edges
    padding_size = int((window_size - 1) / 2)
    x, y = add_padding(x, y, padding_size)

    # scaling z-score values between 0 and 1 to facilitate training
    y = scale_z_scores(y)

    # interpolate missing values (999s)
    # TODO: interpolate missing values
    #  for now this is a dummy function which just writes zeroes
    y = interpolate_values(y)

    # splitting the data into windows of a certain size
    x, y = split_data_into_windows(x, y, window_size)
    print(len(x))
    print(len(y))
    print(x[0].shape)
    print(y[0].shape)

    dataset = DisorderDataset(x, y)

    return dataset


def create_dataframe(path=os.path.join(project_root, "data")):
    
    z_score_path = 'baseline_embeddings_disorder.h5'

    labels_path = 'disorder_labels.fasta'

    x,y = read_data(os.path.join(path, z_score_path), 
                    os.path.join(path, labels_path))

    df = pd.DataFrame({'x': x, 'y': y})

    return df


def collate(batch):
    """
        To be passed to DataLoader as the `collate_fn` argument
    """
    assert isinstance(batch, list)
    data = pad_sequence([b['embeddings'] for b in batch])
    lengths = torch.tensor([len(b['embeddings']) for b in batch])
    padded_batches = pad_sequence([b['z_scores'] for b in batch], padding_value=999.0)
    label = torch.stack([p for p in padded_batches])
    
    return {
        'embeddings': data,
        'z_scores': label,
        'lengths': lengths
    }


def add_padding(embeddings, z_scores, padding_size):
    embeddings = [np.pad(x, ((padding_size, padding_size), (0, 0)), 'constant', constant_values=0) for x in embeddings]
    z_scores = [np.pad(x, padding_size, 'constant', constant_values=999) for x in z_scores]
    return embeddings, z_scores


def scale_z_scores(z_scores):
    return [np.where((x == 999), x + 0, (x + 5) / 21.15) for x in z_scores]


def interpolate_values(z_scores):
    return [np.where((x == 999), 0, x) for x in z_scores]


def split_data_into_windows(embeddings, z_scores, window_size):
    x_new = []
    y_new = []

    for embedding, z_score in zip(embeddings, z_scores):
        z_score_new = np.lib.stride_tricks.sliding_window_view(z_score, window_size)
        embedding_new = np.lib.stride_tricks.sliding_window_view(embedding, window_size, 0)
        for window in z_score_new:
            y_new.append(window)
        for window in embedding_new:
            x_new.append(window)

    return x_new, y_new


# # this is just for testing purposes
# if __name__ == "__main__":
#     load_dataset(path="/home/matthias/Code/Python/pp1cb_ss21/data")
