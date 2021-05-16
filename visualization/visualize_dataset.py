import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#from ..utils.read_embeddings import read_z_scores


def plot_z_score_histogram(z_scores, save_path="./visualization/plots"):
    """
    This method reads in the z-scores and plots their distribution in the form of a histogram.
    Unknown values (999) are removed and reported in the plot title.
    :param save_path: the path where the histogram will be saved
    """
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
    plt.ylabel("count")
    plt.title(f"Distribution of z-scores ({'{:.2f}'.format(unknown_percentage)}% unknown)")
    plt.tight_layout()
    plt.savefig(f"{save_path}/z_score_histogram.png")

    print(f"done! {len(z_scores_flattened)} usable z-scores in total. "
          f"{'{:.2f}'.format(unknown_percentage)}% of all z-scores were unknown and hence removed from the "
          f"histogram.")


def plot_protein_length_histogram(z_scores, save_path="./visualization/plots"):
    """
    This method reads in the z-scores and plots the distribution of protein lengths in the form of a histogram.
    :param save_path: the path where the histogram will be saved
    """
    print("Plotting histogram of the protein lengths...", end="")

    # extracting all protein lengths into a single list
    protein_lengths = [len(list(x)) for x in z_scores.values()]

    # creating the histogram and saving it to a file
    sns.set_style('darkgrid')
    sns.displot(protein_lengths)
    plt.xlabel("protein length")
    plt.ylabel("count")
    plt.title(f"Distribution of protein lengths (median: {int(np.median(protein_lengths))})")
    plt.tight_layout()
    plt.savefig(f"{save_path}/protein_length_histogram.png")

    print(f"done! Median length: {np.median(protein_lengths)}.")


'''if __name__ == "__main__":
    z_scores = read_z_scores(z_score_file="../../data/disorder_labels.fasta")
    plot_z_score_histogram(z_scores)
    plot_protein_length_histogram(z_scores)'''


