import os
import glob
import argparse
import numpy as np


def classification_accuracy(y_true, y_pred):
    y_true = y_true[:y_pred.shape[0], :y_pred.shape[1], :y_pred.shape[2]]

    labels_true = np.argmax(y_true, axis=-1)
    labels_pred = np.argmax(y_pred, axis=-1)

    freespace_label = y_true.shape[-1] - 1
    unkown_label = y_true.shape[-1] - 2

    freespace_mask = labels_true == freespace_label
    unknown_mask = labels_true == unkown_label
    not_unobserved_mask = np.any(y_true > 0.5, axis=-1)
    occupied_mask = ~freespace_mask & ~unknown_mask & not_unobserved_mask
    foreground_mask = ~unknown_mask & not_unobserved_mask

    accuracy = np.mean(
        (labels_true[foreground_mask] ==
         labels_pred[foreground_mask]).astype(np.float32))
    freespace_accuracy = np.mean(
        (labels_true[freespace_mask] ==
         labels_pred[freespace_mask]).astype(np.float32))
    occupied_accuracy = np.mean(
        ((labels_true != unkown_label)[occupied_mask] ==
         (labels_pred != unkown_label)[occupied_mask]).astype(np.float32))
    semantic_accuracy = np.mean(
        (labels_true[occupied_mask] ==
         labels_pred[occupied_mask]).astype(np.float32))

    return accuracy, freespace_accuracy, occupied_accuracy, semantic_accuracy


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scene_path", required=True)
    parser.add_argument("--scene_list_path", required=True)

    return parser.parse_args()


def main():
    args = parse_args()

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
    learned_coarse_to_fine_accuracy = []

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
        learned_coarse_to_fine_path = os.path.join(args.scene_path, scene_name,
                                           "converted/learned-coarse-to-fine-60/probs.npz")

        if not os.path.exists(groundtruth_path) \
                or not os.path.exists(tv_l1_50_path) \
                or not os.path.exists(tv_l1_500_path) \
                or not os.path.exists(learned_full_path) \
                or not os.path.exists(learned_subset_path) \
                or not os.path.exists(learned_coarse_to_fine_path):
            continue

        groundtruth = np.load(groundtruth_path)["probs"]
        tv_l1_50 = np.load(tv_l1_50_path)["probs"]
        tv_l1_500 = np.load(tv_l1_500_path)["probs"]
        learned_full = np.load(learned_full_path)["probs"]
        learned_subset = np.load(learned_subset_path)["probs"]
        learned_coarse_to_fine = np.load(learned_coarse_to_fine_path)["probs"]

        tv_l1_accuracy_50.append(
            classification_accuracy(groundtruth, tv_l1_50))
        tv_l1_accuracy_500.append(
            classification_accuracy(groundtruth, tv_l1_500))
        learned_full_accuracy.append(
            classification_accuracy(groundtruth, learned_full))
        learned_subset_accuracy.append(
            classification_accuracy(groundtruth, learned_subset))
        learned_coarse_to_fine_accuracy.append(
            classification_accuracy(groundtruth, learned_coarse_to_fine))

        print("  TV-L1 (50):", tv_l1_accuracy_50[-1])
        print("  TV-L1 (500):", tv_l1_accuracy_500[-1])
        print("  Learned (full):", learned_full_accuracy[-1])
        print("  Learned (subset):", learned_subset_accuracy[-1])
        print("  Learned (coarse-to-fine):", learned_coarse_to_fine_accuracy[-1])

    tv_l1_accuracy_50 = np.array(tv_l1_accuracy_50)
    tv_l1_accuracy_500 = np.array(tv_l1_accuracy_500)
    learned_full_accuracy = np.array(learned_full_accuracy)
    learned_subset_accuracy = np.array(learned_subset_accuracy)
    learned_coarse_to_fine_accuracy = np.array(learned_coarse_to_fine_accuracy)

    print()
    print("Summary")
    print("  TV-L1 (50):", np.mean(tv_l1_accuracy_50, axis=0))
    print("  TV-L1 (500):", np.mean(tv_l1_accuracy_500, axis=0))
    print("  Learned (full):", np.mean(learned_full_accuracy, axis=0))
    print("  Learned (subset):", np.mean(learned_subset_accuracy, axis=0))
    print("  Learned (coarse-to-fine):", np.mean(learned_coarse_to_fine_accuracy, axis=0))


if __name__ == "__main__":
    main()
