import os
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import numpy as np

from utils.read_embeddings import read_data
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
        self.aa_len = np.array([len(seq) for seq in x])
        self.bins = self.assign_bins(self.y)

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
    y = interpolate_values(y)

    # splitting the data into windows of a certain size
    # the resulting z-score is simply the one in the middle
    x, y = split_data_into_windows(x, y, window_size)
    print(len(x))
    print(len(y))
    print(x[0].shape)
    print(y[0])
    print(min(y))

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
    # there are some labels for which we have values below -5 (e. g. -5.374), so instead of hardcoding 5 and 16.15,
    # we instead take the min and max of the array (these are -5.583)
    # this is not the most elegant solution but we need to exclude 999s for the calculation of the max
    min_total = 0
    max_total = 0
    for z_score_list in z_scores:
        for z_score in z_score_list:
            if z_score < min_total:
                min_total = z_score
            if z_score > max_total and z_score != 999:
                max_total = z_score
    output = [np.where((x == 999), x + 0, (x + abs(min_total)) / (abs(min_total)+max_total)) for x in z_scores]
    return output


def list_duplicates_of(seq, item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at + 1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs


def interpolate_values_in_list(list_of_z_scores, padding_interpolation):
    '''
    This function fist get list of values and find the score, which is 999,
     and replace is with interpolated value
     If padding_interpolation is true, if unknow value(999) is at the beginning and the end of value sequences,
    it will put it into 0. Or else the np.interp will interpolate the value as the first known value
    '''
    number_to_replace = 999
    known_value_list = list_of_z_scores.copy()

    if padding_interpolation:
        for i in range(len(known_value_list)):
            if known_value_list[i] == 999:
                known_value_list[i] = 0
            else:
                break

        for i in range(len(known_value_list)):
            if known_value_list[len(known_value_list) - i - 1] == 999:
                known_value_list[len(known_value_list) - i - 1] = 0
            else:
                break
    end_list = known_value_list
    interpolate_position = list_duplicates_of(known_value_list, number_to_replace)  # find position of value 999
    known_value_list = [i for j, i in enumerate(known_value_list) if
                        j not in interpolate_position]  # get position not with 999
    points = np.arange(0, len(list_of_z_scores), 1).tolist()  # get sequence length

    for pos in interpolate_position:  # remove position with 999
        points.remove(pos)
    interpolate_list = np.interp(interpolate_position, points, known_value_list)
    i = 0  # pointer
    for val in range(len(end_list)):  # put 2 lists together
        if end_list[val] == number_to_replace:
            end_list[val] = interpolate_list[i]
            i = i + 1

    return end_list


def interpolate_values(z_scores):
    return [interpolate_values_in_list(list(x), padding_interpolation=True) for x in z_scores]


def split_data_into_windows(embeddings, z_scores, window_size):
    x_new = []
    y_new = []

    for embedding, z_score in zip(embeddings, z_scores):
        z_score_new = np.lib.stride_tricks.sliding_window_view(z_score, window_size)
        embedding_new = np.lib.stride_tricks.sliding_window_view(embedding, window_size, 0)
        for window in z_score_new:
            # for the z-scores, we only want the middle one
            y_new.append(window[int((window_size-1)/2)])
        for window in embedding_new:
            x_new.append(window)

    return x_new, y_new


# # this is just for testing purposes
# if __name__ == "__main__":
#     dataset = load_dataset(path="/home/matthias/Code/Python/pp1cb_ss21/data", window_size=16)
#     hparams = {'hidden_size': 112,
#                "learning_rate": 1e-4
#                }
#     nested_cross_validation(dataset, mode='evaluate', model=ConvNet(hparams=hparams))

