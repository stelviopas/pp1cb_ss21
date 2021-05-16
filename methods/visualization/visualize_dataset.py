import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# TODO: replace this with import later
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

# ======================= ORIGINAL CODE FROM HERE ON OUT =================================


def plot_z_score_histogram(save_path="plots"):
    """
    This method reads in the z-scores and plots their distribution in the form of a histogram.
    Unknown values (999) are removed and reported in the plot title.
    :param save_path: the path where the histogram will be saved
    """
    z_scores = read_z_scores(z_score_file="../../data/disorder_labels.fasta")

    print("Plotting histogram of all z-scores...", end="")

    # flattening all z-scores to a 1D numpy array
    z_scores_flattened = [list(x) for x in z_scores.values()]
    z_scores_flattened = np.array([item for sublist in z_scores_flattened for item in sublist], dtype=float)

    # counting how often the value 999 appears
    unknown_count = np.count_nonzero(z_scores_flattened == 999)
    unknown_percentage = (unknown_count / len(z_scores_flattened)) * 100

    # removing all 999 values from the z-score array to get a more representative histogram
    z_scores_flattened = np.delete(z_scores_flattened, np.where(z_scores_flattened == 999))

    # creating the histogram and saving it to a file
    sns.set_style('darkgrid')
    sns.displot(z_scores_flattened)
    plt.xlabel("z-score")
    plt.ylabel("density")
    plt.title(f"Distribution of z-scores ({'{:.2f}'.format(unknown_percentage)}% unknown)")
    plt.tight_layout()
    plt.savefig(f"{save_path}/z_score_histogram.png")

    print(f"done! {'{:.2f}'.format(unknown_percentage)}% of all z-scores were unknown and hence removed from the "
          f"histogram.")


if __name__ == "__main__":
    plot_z_score_histogram()
