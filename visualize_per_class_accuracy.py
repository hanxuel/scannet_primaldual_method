import numpy as np
import matplotlib.pyplot as plt


data = np.load("accuracy.npz")

labels = data["labels"]
label_names = data["label_names"][()]
tv_l1_accuracy_50 = data["tv_l1_accuracy_50"]
tv_l1_accuracy_500 = data["tv_l1_accuracy_500"]
learned_full_accuracy = data["learned_full_accuracy"]
learned_subset_accuracy = data["learned_subset_accuracy"]

label_names[19] = "floor mat"

tv_l1_accuracy_50[np.isnan(tv_l1_accuracy_50)] = 0
tv_l1_accuracy_500[np.isnan(tv_l1_accuracy_500)] = 0
learned_full_accuracy[np.isnan(learned_full_accuracy)] = 0
learned_subset_accuracy[np.isnan(learned_subset_accuracy)] = 0

ind = np.arange(len(tv_l1_accuracy_50))

width = 0.3

# plt.style.use("ggplot")

fig, ax = plt.subplots(figsize=(7, 4))

# rects1 = ax.bar(ind, 100 * tv_l1_accuracy_50, width)
rects2 = ax.bar(ind, 100 * tv_l1_accuracy_500, width)
rects3 = ax.bar(ind + width, 100 * learned_subset_accuracy, width)
rects4 = ax.bar(ind + 2 * width, 100 * learned_full_accuracy, width)

# add some text for labels, title and axes ticks
ax.set_ylabel('Reconstruction Accuracy [%]')
ax.set_xticks(ind + 1 * width)
ax.set_xticklabels(list(label_names[label] for label in labels),
                   rotation='vertical')
plt.xlim(-0.5, len(labels))

# ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]),
ax.legend((rects2[0], rects3[0], rects4[0]),
          # ('TV-L1 (50 iters.)', 'TV-L1 (500 iters.)',
          #  'Ours-5 (50 iters.)', 'Ours-300 (50 iters.)'),
          ('TV-L1 (500 iters.)', 'Ours-5 (50 iters.)', 'Ours-300 (50 iters.)'),
          loc='lower left')
plt.tight_layout()
plt.show()
