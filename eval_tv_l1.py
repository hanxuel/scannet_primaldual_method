import os
import glob
import shutil
import argparse
import tempfile
import numpy as np
import tensorflow as tf
import plyfile
import utils
from skimage.measure import marching_cubes_lewiner

#input semantic tsdf output mesh
def mkdir_if_not_exists(path):
    assert os.path.exists(os.path.dirname(path.rstrip("/")))
    if not os.path.exists(path):
        os.makedirs(path)


def write_ply(path, points, color):
    with open(path, "w") as fid:
        fid.write("ply\n")
        fid.write("format ascii 1.0\n")
        fid.write("element vertex {}\n".format(points.shape[0]))
        fid.write("property float x\n")
        fid.write("property float y\n")
        fid.write("property float z\n")
        fid.write("property uchar diffuse_red\n")
        fid.write("property uchar diffuse_green\n")
        fid.write("property uchar diffuse_blue\n")
        fid.write("end_header\n")
        for i in range(points.shape[0]):
            fid.write("{} {} {} {} {} {}\n".format(points[i, 0], points[i, 1],
                                                   points[i, 2], *color))


def weight_variable(name, shape):
    nclasses = shape[-2]

    weight_value = np.zeros([2, 2, 2, nclasses, nclasses, 3], dtype=np.float32)
    for c in range(nclasses):
        weight_value[0, 0, 0, c, c, 0] = -1.
        weight_value[1, 0, 0, c, c, 0] = 1.

        weight_value[0, 0, 0, c, c, 1] = -1.
        weight_value[0, 1, 0, c, c, 1] = 1.

        weight_value[0, 0, 0, c, c, 2] = -1.
        weight_value[0, 0, 1, c, c, 2] = 1.

    weight_value = np.reshape(weight_value, [2, 2, 2, nclasses, 3 * nclasses])

    return tf.constant(weight_value)


def conv3d(value, weights):
    value = tf.pad(
       value,
       [
           [0, 0],
           [0, 1],
           [0, 1],
           [0, 1],
           [0, 0],
       ],
       mode="SYMMETRIC"
    )
    return tf.nn.conv3d(value, weights,
                        strides=[1, 1, 1, 1, 1], padding="VALID")


def conv3d_adj(value, weights, output_shape):
    value = tf.pad(
       value,
       [
           [0, 0],
           [1, 0],
           [1, 0],
           [1, 0],
           [0, 0],
       ],
       mode="SYMMETRIC"
    )

    output_shape[1] += 1
    output_shape[2] += 1
    output_shape[3] += 1

    return tf.nn.conv3d_transpose(value, weights,
                                  output_shape=output_shape,
                                  strides=[1, 1, 1, 1, 1],
                                  padding="SAME")[:, 1:, 1:, 1:, :]


def update_lagrangian(u, l, sig):
    with tf.name_scope("langrange_update"):
        sum_u = tf.reduce_sum(u, axis=4, keep_dims=False)
        return l + sig*(sum_u - 1.0)


def update_dual(u, m, w_shape, sig):
    with tf.name_scope("dual_update"):
        with tf.variable_scope("weights", reuse=True):
            W = weight_variable("w", w_shape)

        _, _, _, _, nclasses = u.get_shape().as_list()
        shape_u = tf.shape(u)
        batch_size = shape_u[0]
        nrows = shape_u[1]
        ncols = shape_u[2]
        nslices = shape_u[3]

        grad_u = conv3d(u, W)

        m += sig * grad_u

        m_rshp = tf.reshape(m, [batch_size, nrows, ncols, nslices, nclasses, 3])

        norm_m = tf.norm(m_rshp, ord="euclidean", axis=5, keep_dims=True)
        norm_m = tf.maximum(norm_m, 1.0)

        m_normalize = tf.divide(m_rshp, tf.tile(norm_m, [1, 1, 1, 1, 1, 3]))

        return tf.reshape(m_normalize, [batch_size, nrows, ncols, nslices,
                                        3 * nclasses])


def update_primal(u, m, l, w_shape, f, tau):
    with tf.name_scope("primal_update"):
        with tf.variable_scope("weights", reuse=True):
            W = weight_variable("w", w_shape)

        _, _, _, _, nclasses = u.get_shape().as_list()
        shape_u = tf.shape(u)
        batch_size = shape_u[0]
        nrows = shape_u[1]
        ncols = shape_u[2]
        nslices = shape_u[3]

        div_m = conv3d_adj(m, W, [batch_size, nrows, ncols, nslices, nclasses])

        l_reshaped = tf.reshape(l, [batch_size, nrows, ncols, nslices, 1])
        u -= tau * (f + tf.tile(l_reshaped, [1, 1, 1, 1, nclasses]) + div_m)

        u = tf.minimum(1.0, tf.maximum(u, 0.0))

        return u


def primal_dual(u, u_, m, l, w_shape, f, sig, tau):
    with tf.name_scope("primal_dual"):
        m = update_dual(u_, m, w_shape, sig)

        l = update_lagrangian(u_, l, sig)

        u_0 = u
        u = update_primal(u, m, l, w_shape, f, tau)
        u_ = 2 * u - u_0

        return u, u_, m, l


def build_model(params):
    nclasses = params["nclasses"]

    # Data cost placeholder.
    d = tf.placeholder(tf.float32, [None, None, None, None, nclasses], name="d")

    # Primal and dual placeholders.
    u  = tf.placeholder(
        tf.float32, [None, None, None, None, nclasses], name="u")
    u_ = tf.placeholder(
        tf.float32, [None, None, None, None, nclasses], name="u_")
    m  = tf.placeholder(
        tf.float32, [None, None, None, None, 3 * nclasses], name="m")
    l  = tf.placeholder(
        tf.float32, [None, None, None, None], name="l")

    w_shape = [2, 2, 2, nclasses, 3 * nclasses]
    with tf.variable_scope("weights"):
        weight_variable("w", w_shape)

    sig = params["sig"]
    tau = params["tau"]
    lam = params["lam"]
    niter = params["niter"]

    d_lam = tf.multiply(d, tf.constant(lam, name="lam"), name="d_lam")
    u_loop = u
    u_loop_= u_
    m_loop = m
    l_loop = l

    for _ in range(niter):
        u_loop, u_loop_, m_loop, l_loop = primal_dual(
            u_loop, u_loop_, m_loop, l_loop, w_shape, d_lam, sig, tau)

    return d, u, u_, m, l, u_loop, u_loop_, m_loop, l_loop


def eval_model(datacost, params):
    batch_size = 1
    nrows = datacost.shape[1]
    ncols = datacost.shape[2]
    nslices = datacost.shape[3]
    nclasses = datacost.shape[4]

    params["nclasses"] = nclasses

    d, u, u_, m, l, u_final, u_final_, m_final, l_final = \
        build_model(params)

    u_init  = np.full([batch_size, nrows, ncols, nslices, nclasses],
                       1.0 / nclasses, dtype=np.float32)
    u_init_ = np.full([batch_size, nrows, ncols, nslices, nclasses],
                       1.0 / nclasses, dtype=np.float32)
    m_init  = np.zeros([batch_size, nrows, ncols, nslices, 3 * nclasses],
                       dtype=np.float32)
    l_init  = np.zeros([batch_size, nrows, ncols, nslices],
                       dtype=np.float32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter("logs")
        writer.add_graph(sess.graph)

        for i in range(params["niter_steps"]):
            print("Iteration", i * params["niter"], "/",
                  params["niter"] * params["niter_steps"])
            (u_init, u_init_, m_init, l_init) = sess.run(
                [u_final, u_final_, m_final, l_final],
                feed_dict={
                    d: datacost,
                    u: u_init[:datacost.shape[0]],
                    u_: u_init_[:datacost.shape[0]],
                    m: m_init[:datacost.shape[0]],
                    l: l_init[:datacost.shape[0]],
                }
            )

    return u_init


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

    with tempfile.NamedTemporaryFile(dir=".", delete=False) as tmpfile:
        plyfile.PlyData([ply_verts, ply_faces]).write(tmpfile.name)
        shutil.move(tmpfile.name, path)
def reduce_resolution(groundtruth,reduce_factor=2):
    volume=np.zeros((groundtruth.shape[0]+2,groundtruth.shape[1]+2,groundtruth.shape[2]+2,groundtruth.shape[3]),dtype=np.float32)
    volume[1:-1,1:-1,1:-1,:]=groundtruth
    volume[:1,1:-1,1:-1,:]=groundtruth[:1,:,:,:]
    volume[-1:,1:-1,1:-1,:]=groundtruth[-1:,:,:,:]
    volume[1:-1,:1,1:-1,:]=groundtruth[:,:1,:,:]
    volume[1:-1,-1:,1:-1,:]=groundtruth[:,-1:,:,:]
    volume[1:-1,1:-1,:1,:]=groundtruth[:,:,:1,:]
    volume[1:-1,1:-1,-1:,:]=groundtruth[:,:,-1:,:]
    reduced_groundtruth=volume[::2,::2,::2,:]
    return np.array(reduced_groundtruth)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datacost_path", default="/home/hliang/Downloads/SUNCG_Data/val_ceil5_wall_5")
    parser.add_argument("--output_path", default="/home/hliang/Downloads/SUNCG_Data/val_ceil5_wall_5/scene34/tv_1")
    parser.add_argument("--label_map_path")

    parser.add_argument("--nclasses", type=int, default=38)
    parser.add_argument("--niter", type=int, default=10)
    parser.add_argument("--niter_steps", type=int, default=10)
    parser.add_argument("--sig", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=100.0)

    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(0)
    tf.set_random_seed(0)

    tf.logging.set_verbosity(tf.logging.INFO)

    params = {
        "niter": args.niter,
        "niter_steps": args.niter_steps,
        "sig": args.sig,
        "tau": args.tau,
        "lam": args.lam,
    }

    label_names = {}
    label_colors = {}
    '''
    if args.datacost_path.endswith(".dat"):
        datacost = utils.read_gvr_datacost(args.datacost_path,
                                           args.nclasses)[None]
        if args.label_map_path:
            with open(args.label_map_path, "r") as fid:
                for line in fid:
                    line = line.strip()
                    if not line:
                        continue
                    label = len(label_names)
                    name = line.split()[0]
                    color = tuple(map(int, line.split()[1:]))
                    label_names[label] = name
                    label_colors[label] = color
    else:
        datacost = np.load(args.datacost_path)["volume"][None]
        assert datacost.shape[-1] == args.nclasses
        if args.label_map_path:
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
                    '''
    Suncg_labels_color=(
        ("ceiling",(0,0,255)),
        ("floor",(255,0,0)),
        ("wall",(0,255,0)),
        ("window",(0,0,44)),
        ("door",(255,26,185)),
        ("chair",(255,211,0)),
        ("bed",(0,88,0)),
        ("sofa",(132,132,255)),
        ("table",(158,79,70)),
        ("side_table",(0,255,193)),
        ("bookshelf",(0,132,149)),
        ("kitchen_cabinet",(0,0,123)),
        ("closets",(149,211,79)),
        ("lamp",(246,158,220)),
        ("car",(211,18,255)),
        ("computer",(123,26,106)),
        ("desk",(246,18,97)),
        ("music",(255,193,132)),
        ("gym",(35,35,9)),
        ("household_appliance",(141,167,123)),
        ("kitchen_appliance",(246,132,9)),
        ("pillow",(132,114,0)),
        ("pets",(114,246,255)),
        ("plants",(158,193,255)),
        ("pool",(114,97,123)),
        ("recreation",(158,0,0)),
        ("night_stand",(0,79,255)),
        ("bathtub",(0,70,149)),
        ("shower",(211,255,0)),
        ("table_and_chair",(185,79,211)),
        ("toilet",(62,0,26)),
        ("trash_can",(237,255,176)),
        ("tvs",(255,123,97)),
        ("sink",(70,255,123)),
        ("mirror",(18,167,97)),
        ("dont_care",(211,167,167)),
        ("unknown",(255,255,255)),
        ("freespace",(0,0,0))
        )
    label_names = tuple(label for label, _ in Suncg_labels_color)
    label_colors = tuple(color for _, color in Suncg_labels_color)
    
    datacost_path=args.datacost_path
    
    groundtruth_path = os.path.join(datacost_path,"scene34","groundtruth.npz")
    groundtruth_data = np.load(groundtruth_path)
    groundtruth=np.zeros(groundtruth_data["shape"],dtype=np.float32)
    groundtruth[groundtruth_data["idxs"][0],groundtruth_data["idxs"][1],groundtruth_data["idxs"][2],groundtruth_data["idxs"][3]]=groundtruth_data["values"]
    groundtruth=reduce_resolution(groundtruth)
    print("groundtruth shape:",groundtruth.shape)
    
    
    datacost_path = os.path.join(datacost_path, "scene34","groundtruth_datacost.npz")
    datacost_data = np.load(datacost_path)
    resolution = datacost_data["resolution"]
    datacost=np.zeros(datacost_data["shape"],dtype=np.float32)
    datacost[datacost_data["idxs"][0],datacost_data["idxs"][1],datacost_data["idxs"][2],datacost_data["idxs"][3]]=datacost_data["values"]
    datacost=reduce_resolution(datacost)
    
    datacost = np.expand_dims(datacost, axis= 0)

    probs = eval_model(datacost, params)[0]
    
    groundtruth = groundtruth[:probs.shape[0], :probs.shape[1], :probs.shape[2]]
    print("groundtruth shape:",groundtruth.shape)

    mkdir_if_not_exists(args.output_path)

    np.savez_compressed(os.path.join(args.output_path, "probs.npz"),
                        probs=probs)

    for label in range(probs.shape[-1]):
        if True:
            path = os.path.join(args.output_path,
                                "{}-{}.ply".format(label, label_names[label]))
            truth_path=os.path.join(args.output_path,
                                "groundtruth{}-{}.ply".format(label, label_names[label]))
            color = label_colors[label]
        else:
            path = os.path.join(args.output_path, "{}.ply".format(label))
            color = None       
        extract_mesh_marching_cubes(path, probs[..., label], color=color)
        extract_mesh_marching_cubes(truth_path, groundtruth[..., label], color=color)

#if __name__ == "__main__":
    #main()
'''
    for label in range(probs.shape[-1]):
        if args.label_map_path:
            path = os.path.join(args.output_path,
                                "{}-{}.ply".format(label, label_names[label]))
            color = label_colors[label]
        else:
            path = os.path.join(args.output_path, "{}.ply".format(label))
            color = None

        extract_mesh_marching_cubes(path, probs[..., label], color=color)
'''
Suncg_labels_color=(
    ("ceiling",(0,0,255)),
    ("floor",(255,0,0)),
    ("wall",(0,255,0)),
    ("window",(0,0,44)),
    ("door",(255,26,185)),
    ("chair",(255,211,0)),
    ("bed",(0,88,0)),
    ("sofa",(132,132,255)),
    ("table",(158,79,70)),
    ("side_table",(0,255,193)),
    ("bookshelf",(0,132,149)),
    ("kitchen_cabinet",(0,0,123)),
    ("closets",(149,211,79)),
    ("lamp",(246,158,220)),
    ("car",(211,18,255)),
    ("computer",(123,26,106)),
    ("desk",(246,18,97)),
    ("music",(255,193,132)),
    ("gym",(35,35,9)),
    ("household_appliance",(141,167,123)),
    ("kitchen_appliance",(246,132,9)),
    ("pillow",(132,114,0)),
    ("pets",(114,246,255)),
    ("plants",(158,193,255)),
    ("pool",(114,97,123)),
    ("recreation",(158,0,0)),
    ("night_stand",(0,79,255)),
    ("bathtub",(0,70,149)),
    ("shower",(211,255,0)),
    ("table_and_chair",(185,79,211)),
    ("toilet",(62,0,26)),
    ("trash_can",(237,255,176)),
    ("tvs",(255,123,97)),
    ("sink",(70,255,123)),
    ("mirror",(18,167,97)),
    ("dont_care",(211,167,167)),
    ("unknown",(255,255,255)),
    ("freespace",(0,0,0))
    )
label_names = tuple(label for label, _ in Suncg_labels_color)
label_colors = tuple(color for _, color in Suncg_labels_color)
data_path="/media/hliang/phenix_xue/Data/Scannet_Data_Code/scannet/scannet_data/scene0019_00"
datacost_path = os.path.join(data_path,"datacost.npz")
groundtruth_path = os.path.join(data_path,"groundtruth.npz")
datacost=np.load(datacost_path)["volume"]
groundtruth=np.load(groundtruth_path)["probs"]
for label in range(groundtruth.shape[-1]):
        if True:
            path = os.path.join(data_path,
                                "{}-{}.ply".format(label, label_names[label]))
            truth_path=os.path.join(data_path,
                                "groundtruth{}-{}.ply".format(label, label_names[label]))
            color = label_colors[label]
        else:
            path = os.path.join(data_path, "{}.ply".format(label))
            color = None       
        extract_mesh_marching_cubes(path, groundtruth[..., label], color=color)
        extract_mesh_marching_cubes(truth_path, groundtruth[..., label], color=color)