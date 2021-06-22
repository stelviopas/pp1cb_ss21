import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 4 different window sizes, 5 folds each = 20 entries per embedding
embeddings = 20 * ['Baseline'] + 20 * ['MSA1'] + 20 * ['MSA2'] + 20 * ['MSA3']
# 4 windows, 4 embeddings = 16
folds = 16 * [1, 2, 3, 4, 5]

# 5 folds each
window_sizes = 5 * [1] + 5 * [7] + 5 * [15] + 5 * [35]
# 4 embeddings
window_sizes = 4 * window_sizes

baseline = [0.0164567, 0.017108, 0.016860, 0.017089697, 0.017097704,
            0.01308, 0.01237, 0.01213, 0.01252, 0.01249,
            0.01157457, 0.01148924, 0.01305872, 0.0122739, 0.01121919,
            0.0114, 0.01128, 0.01087, 0.01101, 0.01201]
MSA1 = [0.01922, 0.01861, 0.01822, 0.01898, 0.01942, 0.01443, 0.01429, 0.01698, 0.01434, 0.01403, 0.01367, 0.01364,
        0.01413, 0.015, 0.01408, 0.01337, 0.01371, 0.01366, 0.01452, 0.01368]
MSA2 = [0.019, 0.01855, 0.01873, 0.01865, 0.01935,
        0.01583, 0.01431, 0.01445, 0.01418, 0.01436,
        0.01383, 0.01364, 0.01383, 0.01515, 0.01501,
        0.0136, 0.01409, 0.01346, 0.0134, 0.01399]
MSA3 = [0.01885, 0.01828, 0.01808, 0.0179, 0.01885,
        0.01468, 0.01434, 0.01373, 0.01431, 0.01439,
        0.01369, 0.01389, 0.01401, 0.01408, 0.01525,
        0.01327, 0.01355, 0.01361, 0.01391, 0.01439]
values = baseline + MSA1 + MSA2 + MSA3

data = {'Embedding': embeddings,
        'Fold': folds,
        'Window Size': window_sizes,
        'MSE': values}

df = pd.DataFrame(data)

sns.set_theme(style="whitegrid")
#ax = sns.barplot(x="Embedding", y="MSE", hue="Window Size", data=df)
ax = sns.barplot(x="Window Size", y="MSE", hue="Embedding", data=df)
plt.legend(title="Embedding", loc="lower right")
sns.despine()
plt.title("Comparison of different window sizes for the embeddings,\nevaluated on validation set")
plt.tight_layout()
plt.savefig("plots/barplot_v4.png", dpi=300)
