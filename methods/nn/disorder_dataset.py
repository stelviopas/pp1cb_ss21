import os
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

from ..utils.read_embeddings import read_data

# TO DO: setup.py

project_root = './../'

# I have specified the default path to a folder named "data" in case 
# no other folder name was provided but the path rather looks cumbersome... 

def create_dataframe(path=os.path.join(project_root, "data")):
    
    z_score_path = 'baseline_embeddings_disorder.h5'

    labels_path = 'disorder_labels.fasta'

    x,y = read_data(os.path.join(path, z_score_path), 
                    os.path.join(path, labels_path))

    df = pd.DataFrame({'x': x, 'y': y})

    return df

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

def calculate_avg_z(y):

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

    return avg_zs





###### TO DO #######

'''
Wrap up (targets, labels) to PyTorch Dataset.
The code below is a basic skeleton for convertion of a data formed as
a list of tuples to a PyTorch dataser.
You might need adapt that for further needs.
'''


class DisorderDataset(Dataset):

    def __init__(self, data):

        """ 
        In: 
            data: list of tuples ()
        """

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        """
        In:
            i: an integer value to index data
        Outs:
            data: A dictionary of {data, label}
            """
        _, _, indices, label = self.data[i]

        return {
            'data': torch.tensor(indices).long(),
            'label': torch.tensor(label).float()
        }

###### END TO DO ########