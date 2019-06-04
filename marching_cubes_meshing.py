import os
import argparse
import numpy as np
import plyfile
from skimage.measure import marching_cubes_lewiner


def extract_mesh_marching_cubes(path, volume, color=None, level=0.5,
                                step_size=1.0, gradient_direction="ascent"):
    if level > volume.max() or level < volume.min():
        return

    verts, faces, normals, values = marching_cubes_lewiner(
        volume, level=level, step_size=step_size,
        gradient_direction=gradient_direction)

    ply_verts = np.empty(len(verts),
                         dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    ply_verts["x"] = verts[:, 0]
    ply_verts["y"] = verts[:, 1]
    ply_verts["z"] = verts[:, 2]
    ply_verts = plyfile.PlyElement.describe(ply_verts, "vertex")

    if color is None:
        ply_faces = np.empty(
            len(faces), dtype=[("vertex_indices", "i4", (3,))])
    else:
        ply_faces = np.empty(
            len(faces), dtype=[("vertex_indices", "i4", (3,)),
                               ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        ply_faces["red"] = color[0]
        ply_faces["green"] = color[1]
        ply_faces["blue"] = color[2]
    ply_faces["vertex_indices"] = faces
    ply_faces = plyfile.PlyElement.describe(ply_faces, "face")

    plyfile.PlyData([ply_verts, ply_faces]).write(path)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prob_path", required=True)
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--label_map_path")

    return parser.parse_args()


def main():
    args = parse_args()

    assert os.path.isdir(args.mesh_path)

    prob = np.load(args.prob_path)["probs"]

    if args.label_map_path:
        label_names = {}
        label_colors = {}
        with open(args.label_map_path, "r") as fid:
            for line in fid:
                line = line.strip()
                if not line:
                    continue
                label = int(line.split(":")[0].split()[0])
                name = " ".join(line.split(":")[0].split()[1:])
                color = tuple(map(int, line.split(":")[1].split()))
                label_names[label] = name
                label_colors[label] = color

    for label in range(prob.shape[-1]):
        if args.label_map_path:
            path = os.path.join(args.mesh_path,
                                "{}-{}.ply".format(label, label_names[label]))
            color = label_colors[label]
        else:
            path = os.path.join(args.mesh_path, "{}.ply".format(label))
            color = None

        extract_mesh_marching_cubes(path, prob[..., label], color=color)


if __name__ == "__main__":
    main()
