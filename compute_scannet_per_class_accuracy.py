import os
import glob
import argparse
import numpy as np


def classification_accuracy(y_true, y_pred, labels):
    y_true = y_true[:y_pred.shape[0], :y_pred.shape[1], :y_pred.shape[2]]

    labels_true = np.argmax(y_true, axis=-1)
    labels_pred = np.argmax(y_pred, axis=-1)
    not_unobserved_mask = np.any(y_true > 0.5, axis=-1)

    accuracy = []
    count = []
    for label in labels:
        mask = (labels_true == label) & not_unobserved_mask
        accuracy.append(np.sum((labels_pred[mask] == label).astype(np.float32)))
        count.append(np.sum(mask))

    return accuracy, count


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scene_path", required=True)
    parser.add_argument("--scene_list_path", required=True)
    parser.add_argument("--label_map_path", required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    labels = []
    label_names = {}
    label_colors = {}
    with open(args.label_map_path, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line:
                continue
            label = int(line.split(":")[0].split()[0])
            name = line.split(":")[0].split()[1]
            color = tuple(map(int, line.split(":")[1].split()))
            label_names[label] = name
            label_colors[label] = color
            labels.append(label)

    scene_list = []
    with open(args.scene_list_path, "r") as fid:
        for line in fid:
            line = line.strip()
            if line:
                scene_list.append(line)

    tv_l1_accuracy_50 = []
    tv_l1_accuracy_500 = []
    learned_full_accuracy = []
    learned_subset_accuracy = []
    tv_l1_count_50 = []
    tv_l1_count_500 = []
    learned_full_count = []
    learned_subset_count = []

    for i, scene_name in enumerate(scene_list):
        print("Processing {} [{}/{}]".format(scene_name, i + 1, len(scene_list)))

        groundtruth_path = os.path.join(args.scene_path, scene_name,
                                        "converted/groundtruth_model/probs.npz")
        tv_l1_50_path = os.path.join(args.scene_path, scene_name,
                                     "converted/tv_l1_model-50/probs.npz")
        tv_l1_500_path = os.path.join(args.scene_path, scene_name,
                                      "converted/tv_l1_model-500/probs.npz")
        learned_full_path = os.path.join(args.scene_path, scene_name,
                                         "converted/learned-60/probs.npz")
        learned_subset_path = os.path.join(args.scene_path, scene_name,
                                           "converted/learned-subset-60/probs.npz")

        if not os.path.exists(groundtruth_path) \
                or not os.path.exists(tv_l1_50_path) \
                or not os.path.exists(tv_l1_500_path) \
                or not os.path.exists(learned_full_path) \
                or not os.path.exists(learned_subset_path):
            continue

        groundtruth = np.load(groundtruth_path)["probs"]
        tv_l1_50 = np.load(tv_l1_50_path)["probs"]
        tv_l1_500 = np.load(tv_l1_500_path)["probs"]
        learned_full = np.load(learned_full_path)["probs"]
        learned_subset = np.load(learned_subset_path)["probs"]

        for label in sorted(label_names):
            accuracy, count = \
                classification_accuracy(groundtruth, tv_l1_50, labels)
            tv_l1_accuracy_50.append(accuracy)
            tv_l1_count_50.append(count)
            accuracy, count = \
                classification_accuracy(groundtruth, tv_l1_500, labels)
            tv_l1_accuracy_500.append(accuracy)
            tv_l1_count_500.append(count)
            accuracy, count = \
                classification_accuracy(groundtruth, learned_full, labels)
            learned_full_accuracy.append(accuracy)
            learned_full_count.append(count)
            accuracy, count = \
                classification_accuracy(groundtruth, learned_subset, labels)
            learned_subset_accuracy.append(accuracy)
            learned_subset_count.append(count)

        print("  TV-L1 (50):", tv_l1_accuracy_50[-1])
        print("  TV-L1 (500):", tv_l1_accuracy_500[-1])
        print("  Learned (full):", learned_full_accuracy[-1])
        print("  Learned (subset):", learned_subset_accuracy[-1])

    tv_l1_accuracy_50 = np.array(tv_l1_accuracy_50)
    tv_l1_accuracy_500 = np.array(tv_l1_accuracy_500)
    learned_full_accuracy = np.array(learned_full_accuracy)
    learned_subset_accuracy = np.array(learned_subset_accuracy)
    tv_l1_count_50 = np.array(tv_l1_count_50)
    tv_l1_count_500 = np.array(tv_l1_count_500)
    learned_full_count = np.array(learned_full_count)
    learned_subset_count = np.array(learned_subset_count)

    tv_l1_accuracy_50 = \
        np.sum(tv_l1_accuracy_50, axis=0) / np.sum(tv_l1_count_50, axis=0)
    tv_l1_accuracy_500 = \
        np.sum(tv_l1_accuracy_500, axis=0) / np.sum(tv_l1_count_500, axis=0)
    learned_full_accuracy = \
        np.sum(learned_full_accuracy, axis=0) / np.sum(learned_full_count, axis=0)
    learned_subset_accuracy = \
        np.sum(learned_subset_accuracy, axis=0) / np.sum(learned_subset_count, axis=0)

    np.savez("accuracy.npz",
             tv_l1_accuracy_50=tv_l1_accuracy_50,
             tv_l1_accuracy_500=tv_l1_accuracy_500,
             learned_full_accuracy=learned_full_accuracy,
             learned_subset_accuracy=learned_subset_accuracy,
             labels=labels, label_names=label_names, label_colors=label_colors)


if __name__ == "__main__":
    main()
