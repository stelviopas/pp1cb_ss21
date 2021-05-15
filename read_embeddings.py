import h5py
import numpy as np


def read_embeddings(embedding_file):
    print("Reading embeddings...", end="")

    embeddings = dict()
    embeddings_list = []
    with h5py.File(embedding_file, 'r') as f:
        for key in f.keys():
            embeddings[key] = np.array(f[key], dtype=np.float32)
            embeddings_list.append(np.array(f[key], dtype=np.float32))

    embeddings_arr = np.concatenate(embeddings_list)
    print(f"done! Found {len(embeddings.keys())} proteins.")
    return embeddings


def read_z_scores(z_score_file):
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
                z_scores[protein_id] = [float(x) for x in line[1:-1].split(", ")]

    print(f"done! Found {len(z_scores.keys())} proteins.")
    return z_scores


if __name__ == "__main__":
    # TODO: we can later use the embeddings_arr to split the arrays directly
    embeddings = read_embeddings("data/baseline_embeddings_disorder.h5")
    z_scores = read_z_scores("data/disorder_labels.fasta")

    # sanity check that we have the same proteins in both sets
    assert set(embeddings.keys()) == set(z_scores.keys()), "Protein IDs between embeddings and Z-scores do not match."
