import os
import re 
import pickle 
import random 

import torch
from torch.utils.data.dataset import Dataset

from read_embeddings import *
from read_fasta import *


###### TO DO #######

''' Wrap up to PyTorch Dataset'''

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

 def create_dataframe(path="/home/anastasiaaa/uni/pp1cb_ss21/dataset"):
 	
 	z_score_path = 'baseline_embeddings_disorder.h5'

 	labels_path = 'disorder_labels.fasta'

 	df = read_data(os.join.path(path, z_score_path), 
 					os.join.path(path, labels_path))

 	return df
