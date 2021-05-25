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

            #avg_zs.append(total_z_sum/trimmed_length)
            
            avg_zs.append(total_z_sum)
            
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

    


def load_dataset(path=os.path.join(project_root, "data")):

    z_score_path = 'baseline_embeddings_disorder.h5'

    labels_path = 'disorder_labels.fasta'

    x,y = read_data(os.path.join(path, z_score_path), 
                    os.path.join(path, labels_path))
    new_x = []
    
    test = 23
    
    for protein in x:
        new_x.append(protein[0:test])
    x = new_x

    new_y = []
    
    for z_score in y:
        new_y.append(z_score[0:test])
    y = new_y
    
    
    dataset = DisorderDataset(x,y)

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
    data = pad_sequence([b['embeddings'] for b in batch], batch_first = True)
    lengths = torch.tensor([len(b['embeddings']) for b in batch])
    padded_batches = pad_sequence([b['z_scores'] for b in batch], padding_value=999.0, batch_first = True)
    label = torch.stack([p for p in padded_batches])
    
    print(f"Size of padded data {data.size()}")
    return {
        'embeddings': data,
        'z_scores': label,
        'lengths': lengths
    }