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



###### TO DO #######

''' Wrap up (targets, labels) to PyTorch Dataset.
The code below is a basic skeleton for convertion of a data formed as a list of tuples to
a PyTorch dataser. You might need adapt that for further needs. '''

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