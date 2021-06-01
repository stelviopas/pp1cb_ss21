import math
import os

import h5py
import numpy as np


def read_embeddings(embedding_file):
    """
    Reads in an embedding file in h5 format.
    :param embedding_file: the path to the h5 file
    :return: dictionary with protein IDs as keys and embeddings (2D numpy arrays) as values
    """
    print("Reading embeddings...", end="")

    embeddings = dict()
    with h5py.File(embedding_file, 'r') as f:
        for key in f.keys():
            embeddings[key] = np.array(f[key], dtype=np.float32)

    print(f"done! Found {len(embeddings.keys())} proteins.")

    return embeddings


def read_z_scores(z_score_file):
    """
    Reads in a file in fasta format.
    :param z_score_file: path to the fasta file
    :return: dictionary with protein IDs as keys and z-scores (1D numpy arrays) as values
    """
    print("Reading Z-scores...", end="")

    z_scores = dict()
    protein_id = ""
    with open(z_score_file, 'r', newline='') as reader:
        for line in reader.readlines():
            line = line.rstrip()
            if line.startswith('>'):
                protein_id = line[1:]
            else:
                # removing the [ and ] at the end of the line and splitting the data into a list
                z_scores[protein_id] = np.array([float(x) for x in line[1:-1].split(", ")])

    print(f"done! Found {len(z_scores.keys())} proteins.")

    return z_scores


def match_data(embeddings, z_scores):
    """
    Takes two dictionaries (protein IDs as keys, need to be identical) and matches them according to the IDs.
    Returns two lists x and y.
    These can not be converted to np arrays because different proteins have a different number of amino acids.
    :param embeddings: embedding dict
    :param z_scores: z_score dict
    :return: list containing embeddings, array containing z-scores, the order of proteins is identical in both
    """
    print("Matching protein IDs...", end="")

    x, y = [], []
    # this could just as well be the keys from the z-scores, since the two of them are identical
    protein_ids = embeddings.keys()
    for protein_id in protein_ids:
        x.append(embeddings[protein_id])
        y.append(z_scores[protein_id])

    print(f"done!")
    return x, y


def read_data(embedding_path, z_score_path):
    """
    Reading in the data and matching it using the protein IDs. Finally, two np arrays are returned.
    Since matching the data takes some time, we are able to cache the two resulting arrays, so that this only has
    to be done once.
    :param embedding_path: path to the embedding file
    :param z_score_path: path to the z_score file
    :return: (x: embeddings, y: z-scores)
    """
    # reading in the two necessary files
    embeddings = read_embeddings(embedding_path)
    z_scores = read_z_scores(z_score_path)

    # sanity check that we have the same proteins in both sets
    assert set(embeddings.keys()) == set(z_scores.keys()), "Protein IDs between embeddings and Z-scores do not match."

    # combining the data into a single dataframe
    x, y = match_data(embeddings, z_scores)

    return x, y


def split_data(y, num_folds=10):
    """
    Takes two lists (x = embeddings, y = z-scores) and splits the data into ten folds.
    Returns a dictionary where the key is the fold ID (0, 1, 2, ...) and the values are the indices.
    This method does not stratify folds and does not care about potential class imbalances.
    Such checks should be added!
    :param y: z-score list
    :param num_folds: the number of folds into which the data is split, default 10
    :return: dictionary with keys as fold IDs and indices as values
    """
    print(f"Creating splits...", end="")
    fold_dict = dict()
    start_index = 0
    # if the number of proteins is not evenly divisible by the number of folds, the last samples are distributed
    # evenly across folds
    fold_size = math.floor(len(y) / num_folds)
    for fold in range(num_folds):
        fold_dict[fold] = list(range(start_index, start_index + fold_size))
        start_index += fold_size

    # distributing samples which are left over (due to the number of samples not being divisible by the number of folds)
    # evenly across folds
    fold = 0
    while start_index < len(y):
        fold_dict[fold] += [start_index]
        start_index += 1
        fold += 1

    # sanity check that we did not loose any samples while splitting
    assert sum([len(fold) for fold in fold_dict.values()]) == len(y), "Number of samples after splitting does not " \
                                                                      "match number of samples before splitting."

    additional_text = "" if len(y) % num_folds == 0 else f" with {len(y) % num_folds} left over samples " \
                                                         f"being distributed evenly among folds"
    print(f"done! Created {num_folds} splits of size {fold_size}{additional_text}.")

    return fold_dict





# if __name__ == "__main__":
#     x, y = read_data(embedding_path="../../data/baseline_embeddings_disorder.h5", z_score_path="../../data/disorder_labels.fasta")
