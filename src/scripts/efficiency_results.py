import numpy as np

base_url = ""

n_mentions = []
n_words = []
time_md = []
time_ed = []

with open(
    "{}/generated/efficiency_gpu.txt".format(base_url), "r", encoding="utf-8"
) as f:
    for line in f:
        splt = line.split()

        n_words.append(int(splt[0]))
        n_mentions.append(int(splt[1]))
        time_md.append(float(splt[2]))
        time_ed.append(float(splt[3]))


print("statistics words", np.round(np.mean(n_words)), np.round(np.std(n_words)))
print(
    "statistics mentions found",
    np.round(np.mean(n_mentions)),
    np.round(np.std(n_mentions)),
)
print(
    "statistics time md",
    np.round(np.mean(time_md), 4),
    np.round(np.std(time_md), 4),
    np.sum(time_md),
)
print(
    "statistics time ed",
    np.round(np.mean(time_ed), 4),
    np.round(np.std(time_ed), 4),
    np.sum(time_ed),
)
