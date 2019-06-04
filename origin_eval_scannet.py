import os
import glob
import shutil
import argparse
import tempfile
import numpy as np
import tensorflow as tf
import plyfile
from skimage.measure import marching_cubes_lewiner


def mkdir_if_not_exists(path):
    assert os.path.exists(os.path.dirname(path.rstrip("/")))
    if not os.path.exists(path):
        os.makedirs(path)


def conv_weight_variable(name, shape, stddev=1.0):
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    return tf.get_variable(name, shape, dtype=tf.float32,
                           initializer=initializer)


def bias_weight_variable(name, shape, cval=0.0):
    initializer = tf.constant_initializer(cval)
    return tf.get_variable(name, shape, dtype=tf.float32,
                           initializer=initializer)


def conv3d(x, weights, name=None):
    return tf.nn.conv3d(x, weights, name=name,
                        strides=[1, 1, 1, 1, 1], padding="SAME")


def conv3d_adj(x, weights, num_channels, name=None):
    output_shape = x.get_shape().as_list()
    output_shape[0] = tf.shape(x)[0]
    output_shape[4] = num_channels
    return tf.nn.conv3d_transpose(x, weights, name=name,
                                  output_shape=output_shape,
                                  strides=[1, 1, 1, 1, 1], padding="SAME")


def avg_pool3d(x, factor, name=None):
    return tf.nn.avg_pool3d(x,
                            ksize=[1, factor, factor, factor, 1],
                            strides=[1, factor, factor, factor, 1],
                            padding="SAME", name=name)


def max_pool3d(x, factor, name=None):
    return tf.nn.max_pool3d(x,
                            ksize=[1, factor, factor, factor, 1],
                            strides=[1, factor, factor, factor, 1],
                            padding="SAME", name=name)


def resize_volumes(x, row_factor, col_factor, slice_factor):
    output = repeat_elements(x, row_factor, axis=1)
    output = repeat_elements(output, col_factor, axis=2)
    output = repeat_elements(output, slice_factor, axis=3)
    return output


def concatenate(tensors, axis=-1):
    """Concatenates a list of tensors alongside the specified axis.
    # Arguments
        tensors: list of tensors to concatenate.
        axis: concatenation axis.
    # Returns
        A tensor.
    """
    if axis < 0:
        rank = ndim(tensors[0])
        if rank:
            axis %= rank
        else:
            axis = 0

    return tf.concat([x for x in tensors], axis)


def repeat_elements(x, rep, axis):
    """Repeats the elements of a tensor along an axis, like `np.repeat`.
    If `x` has shape `(s1, s2, s3)` and `axis` is `1`, the output
    will have shape `(s1, s2 * rep, s3)`.
    # Arguments
        x: Tensor or variable.
        rep: Python integer, number of times to repeat.
        axis: Axis along which to repeat.
    # Returns
        A tensor.
    """
    x_shape = x.get_shape().as_list()
    # For static axis
    if x_shape[axis] is not None:
        # slices along the repeat axis
        splits = tf.split(value=x, num_or_size_splits=x_shape[axis], axis=axis)
        # repeat each slice the given number of reps
        x_rep = [s for s in splits for _ in range(rep)]
        return concatenate(x_rep, axis)

    # Here we use tf.tile to mimic behavior of np.repeat so that
    # we can handle dynamic shapes (that include None).
    # To do that, we need an auxiliary axis to repeat elements along
    # it and then merge them along the desired axis.

    # Repeating
    auxiliary_axis = axis + 1
    x_shape = tf.shape(x)
    x_rep = tf.expand_dims(x, axis=auxiliary_axis)
    reps = np.ones(len(x.get_shape()) + 1)
    reps[auxiliary_axis] = rep
    x_rep = tf.tile(x_rep, reps)

    # Merging
    reps = np.delete(reps, auxiliary_axis)
    reps[axis] = rep
    reps = tf.constant(reps, dtype='int32')
    x_shape = x_shape * reps
    x_rep = tf.reshape(x_rep, x_shape)

    # Fix shape representation
    x_shape = x.get_shape().as_list()
    x_rep.set_shape(x_shape)
    x_rep._keras_shape = tuple(x_shape)

    return x_rep


def update_lagrangian(u_, l, sig, level):
    assert len(u_) == len(l)

    with tf.name_scope("lagrange_update"):
        sum_u = tf.reduce_sum(u_[level], axis=4, keep_dims=False)
        l[level] += sig * (sum_u - 1.0)


def update_dual(u_, m, sig, level):
    assert len(u_) == len(m)

    with tf.name_scope("dual_update"):
        with tf.variable_scope("weights{}".format(level), reuse=True):
            w1 = tf.get_variable("w1")
            w2 = tf.get_variable("w2")

        _, nrows, ncols, nslices, nclasses = \
            u_[level].get_shape().as_list()
        batch_size = tf.shape(u_[level])[0]

        if level + 1 < len(u_):
            grad_u1 = conv3d(u_[level], w1)
            grad_u2 = conv3d(u_[level + 1], w2)
            grad_u = grad_u1 + resize_volumes(grad_u2, 2, 2, 2)
        else:
            grad_u = conv3d(u_[level], w1)

        m[level] += sig * grad_u

        m_rshp = tf.reshape(m[level], [batch_size, nrows, ncols,
                                       nslices, nclasses, 3])

        m_norm = tf.norm(m_rshp, ord="euclidean", axis=5, keep_dims=True)
        m_norm = tf.maximum(m_norm, 1.0)
        m_normalized = tf.divide(m_rshp, m_norm)

        m[level] = tf.reshape(m_normalized, [batch_size, nrows, ncols,
                                             nslices, nclasses * 3])


def update_primal(u, m, l, d, tau, level):
    assert len(u) == len(m)
    assert len(u) == len(l)
    assert len(u) == len(d)

    with tf.name_scope("primal_update"):
        with tf.variable_scope("weights{}".format(level), reuse=True):
            w1 = tf.get_variable("w1")
            w2 = tf.get_variable("w2")

        # u_shape = tf.shape(u[level])
        # batch_size = u_shape[0]
        # nrows = u_shape[1]
        # ncols = u_shape[2]
        # nslices = u_shape[3]
        # nclasses = u_shape[4]

        _, nrows, ncols, nslices, nclasses = \
            u[level].get_shape().as_list()
        batch_size = tf.shape(u[level])[0]

        if level + 1 < len(u):
            div_m1 = conv3d_adj(m[level], w1, nclasses)
            div_m2 = conv3d_adj(m[level + 1], w2, nclasses)
            div_m = div_m1 + resize_volumes(div_m2, 2, 2, 2)
        else:
            div_m = conv3d_adj(m[level], w1, nclasses)

        l_rshp = tf.reshape(l[level], [batch_size, nrows, ncols, nslices, 1])

        u[level] -= tau * (d[level] + l_rshp + div_m)

        u[level] = tf.minimum(1.0, tf.maximum(u[level], 0.0))


def primal_dual(u, u_, m, l, d, sig, tau, iter):
    u_0 = list(u)

    for level in list(range(len(u)))[::-1]:
        with tf.name_scope("primal_dual_iter{}_level{}".format(iter, level)):
            update_dual(u_, m, sig, level)

            update_lagrangian(u_, l, sig, level)

            update_primal(u, m, l, d, tau, level)

            u_[level] = 2 * u[level] - u_0[level]

    return u, u_, m, l


def classification_accuracy(y_true, y_pred):
    with tf.name_scope("classification_accuracy"):
        labels_true = tf.argmax(y_true, axis=-1)
        labels_pred = tf.argmax(y_pred, axis=-1)

        freespace_label = y_true.shape[-1] - 1
        unkown_label = y_true.shape[-1] - 2

        freespace_mask = tf.equal(labels_true, freespace_label)
        unknown_mask = tf.equal(labels_true, unkown_label)
        not_unobserved_mask = tf.reduce_any(tf.greater(y_true, 0.5), axis=-1)
        occupied_mask = ~freespace_mask & ~unknown_mask & not_unobserved_mask

        freespace_accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.boolean_mask(labels_true, freespace_mask),
            tf.boolean_mask(labels_pred, freespace_mask)), dtype=tf.float32))
        occupied_accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.boolean_mask(tf.less(labels_true, freespace_label),
                            occupied_mask),
            tf.boolean_mask(tf.less(labels_pred, freespace_label),
                            occupied_mask)), dtype=tf.float32))
        semantic_accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.boolean_mask(labels_true, occupied_mask),
            tf.boolean_mask(labels_pred, occupied_mask)), dtype=tf.float32))

    return freespace_accuracy, occupied_accuracy, semantic_accuracy


def categorical_crossentropy(y_true, y_pred, params):
    nclasses = y_true.shape[-1]

    with tf.name_scope("categorical_cross_entropy"):
        y_true = tf.nn.softmax(params["softmax_scale"] * y_true)
        # y_pred = tf.nn.softmax(params["softmax_scale"] * y_pred)

        y_true = tf.clip_by_value(y_true, EPSILON, 1.0 - EPSILON)
        y_pred = tf.clip_by_value(y_pred, EPSILON, 1.0 - EPSILON)

        # Measure how close we are to unknown class.
        unkown_weights = 1 - y_true[..., -2][..., None]

        # Measure how close we are to unobserved class by measuring KL
        # divergence to uniform distribution.
        unobserved_weights = tf.log(tf.cast(nclasses, tf.float32)) + \
                             tf.reduce_sum(y_true * tf.log(y_true),
                                           axis=-1, keep_dims=True)

        # Final per voxel loss function weights.
        weights = tf.maximum(EPSILON, unkown_weights * unobserved_weights)

        # Compute weighted cross entropy.
        cross_entropy = -tf.reduce_sum(weights * y_true * tf.log(y_pred)) / \
                         tf.reduce_sum(weights)

    return cross_entropy


def build_model(params):
    batch_size = params["batch_size"]
    nlevels = params["nlevels"]
    nrows = params["nrows"]
    ncols = params["ncols"]
    nslices = params["nslices"]
    nclasses = params["nclasses"]
    softmax_scale = params["softmax_scale"]

    # Setup placeholders and variables.

    d = tf.placeholder(tf.float32, [None, nrows, ncols,
                                    nslices, nclasses], name="d")

    u = []
    u_ = []
    m = []
    l = []
    for level in range(nlevels):
        factor = 2 ** level
        assert nrows % factor == 0
        assert ncols % factor == 0
        assert nslices % factor == 0
        nrows_level = nrows // factor
        ncols_level = ncols // factor
        nslices_level = nslices // factor
        u.append(tf.placeholder(
                     tf.float32, [None, nrows_level, ncols_level,
                                  nslices_level, nclasses],
                     name="u{}".format(level)))
        u_.append(tf.placeholder(
                      tf.float32, [None, nrows_level, ncols_level,
                                   nslices_level, nclasses],
                     name="u_{}".format(level)))
        m.append(tf.placeholder(
                     tf.float32, [None, nrows_level, ncols_level,
                                  nslices_level, 3 * nclasses],
                     name="m{}".format(level)))
        l.append(tf.placeholder(
                     tf.float32, [None, nrows_level, ncols_level,
                                  nslices_level],
                     name="l{}".format(level)))

        with tf.variable_scope("weights{}".format(level)):
            conv_weight_variable(
                "w1", [2, 2, 2, nclasses, 3 * nclasses], stddev=0.001)
            conv_weight_variable(
                "w2", [2, 2, 2, nclasses, 3 * nclasses], stddev=0.001)

    sig = params["sig"]
    tau = params["tau"]
    lam = params["lam"]
    niter = params["niter"]

    d_lam = tf.multiply(d, lam, name="d_lam")

    d_encoded = []
    for level in range(nlevels):
        with tf.name_scope("datacost_encoding{}".format(level)):
            if level > 0:
                d_lam = avg_pool3d(d_encoded[level - 1], 2)

            with tf.variable_scope("weights{}".format(level)):
                w1_d = conv_weight_variable(
                    "w1_d", [5, 5, 5, nclasses, nclasses], stddev=0.01*lam)
                w2_d = conv_weight_variable(
                    "w2_d", [5, 5, 5, nclasses, nclasses], stddev=0.01*lam)
                w3_d = conv_weight_variable(
                    "w3_d", [5, 5, 5, nclasses, nclasses], stddev=0.01*lam)
                b1_d = bias_weight_variable("b1_d", [nclasses])
                b2_d = bias_weight_variable("b2_d", [nclasses])
                b3_d = bias_weight_variable("b3_d", [nclasses])

            d_residual = conv3d(d_lam, w1_d)
            d_residual = tf.nn.relu(d_residual + b1_d)
            d_residual = conv3d(d_residual, w2_d)
            d_residual = tf.nn.relu(d_residual + b2_d)
            d_residual = conv3d(d_residual, w3_d,
                                name="d_encoded{}".format(level))
            d_residual += b3_d

            d_encoded.append(d_lam + d_residual)

    # Create a copy of the placeholders for the loop variables.
    u_loop = list(u)
    u_loop_= list(u_)
    m_loop = list(m)
    l_loop = list(l)

    for iter in range(niter):
        u_loop, u_loop_, m_loop, l_loop = primal_dual(
            u_loop, u_loop_, m_loop, l_loop, d_encoded, sig, tau, iter)

    probs = u_loop

    for level in range(nlevels):
        u_loop[level] = tf.identity(
            u_loop[level], name="u_final{}".format(level))
        u_loop_[level] = tf.identity(
            u_loop_[level], name="u_final_{}".format(level))
        m_loop[level] = tf.identity(
            m_loop[level], name="m_final{}".format(level))
        l_loop[level] = tf.identity(
            l_loop[level], name="l_final{}".format(level))

    for level in range(nlevels):
        with tf.name_scope("prob_decoding{}".format(level)):
            with tf.variable_scope("weights{}".format(level)):
                w1_p = conv_weight_variable(
                    "w1_p", [5, 5, 5, nclasses, nclasses],
                    stddev=0.01*softmax_scale)
                w2_p = conv_weight_variable(
                    "w2_p", [5, 5, 5, nclasses, nclasses],
                    stddev=0.01*softmax_scale)
                w3_p = conv_weight_variable(
                    "w3_p", [5, 5, 5, nclasses, nclasses],
                    stddev=0.01*softmax_scale)
                b1_p = bias_weight_variable("b1_p", [nclasses])
                b2_p = bias_weight_variable("b2_p", [nclasses])
                b3_p = bias_weight_variable("b3_p", [nclasses])

            probs_residual = conv3d(probs[level], w1_p)
            probs_residual = tf.nn.relu(probs_residual + b1_p)
            probs_residual = conv3d(probs_residual, w2_p)
            probs_residual = tf.nn.relu(probs_residual + b2_p)
            probs_residual = conv3d(probs_residual, w3_p)
            probs_residual += b3_p

        probs[level] = tf.nn.softmax(softmax_scale * probs[level]
                                     + probs_residual,
                                     name="probs{}".format(level))

    return probs, d, u, u_, m, l


def eval_model(checkpoint_path, datacost_path, params):
    niter_steps = params["niter_steps"]
    nlevels = params["nlevels"]

    datacost = np.load(datacost_path)
    resolution = datacost["resolution"]
    datacost = datacost["volume"]

    orig_shape = datacost.shape
    datacost = datacost[:datacost.shape[0]-(datacost.shape[0]%(2**(nlevels-1))),
                        :datacost.shape[1]-(datacost.shape[1]%(2**(nlevels-1))),
                        :datacost.shape[2]-(datacost.shape[2]%(2**(nlevels-1)))]

    print("Cropping datacost from", orig_shape, "to", datacost.shape)

    nrows = datacost.shape[0]
    ncols = datacost.shape[1]
    nslices = datacost.shape[2]
    nclasses = datacost.shape[3]

    params["nrows"] = nrows
    params["ncols"] = ncols
    params["nslices"] = nslices

    print("Building model")
    build_model(params)

    with tf.Session() as sess:
        # print("Reading meta graph")
        # saver = tf.train.import_meta_graph(
        #     os.path.join(checkpoint_path, "initial.meta"))

        print("Reading checkpoint")
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

        print("Initializing variables")

        graph = tf.get_default_graph()

        u_init = []
        u_init_ = []
        m_init = []
        l_init = []
        for level in range(nlevels):
            factor = 2 ** level
            assert nrows % factor == 0
            assert ncols % factor == 0
            assert nslices % factor == 0
            nrows_level = nrows // factor
            ncols_level = ncols // factor
            nslices_level = nslices // factor
            u_init.append(np.full([1, nrows_level, ncols_level,
                                   nslices_level, nclasses], 1.0 / nclasses,
                                   dtype=np.float32))
            u_init_.append(np.full([1, nrows_level, ncols_level,
                                    nslices_level, nclasses], 1.0 / nclasses,
                                    dtype=np.float32))
            m_init.append(np.zeros([1, nrows_level, ncols_level,
                                    nslices_level, 3 * nclasses],
                                   dtype=np.float32))
            l_init.append(np.zeros([1, nrows_level, ncols_level,
                                    nslices_level],
                                   dtype=np.float32))

        d = graph.get_tensor_by_name("d:0")
        p = graph.get_tensor_by_name("probs0:0")

        u = []
        u_ = []
        m = []
        l = []
        u_final = []
        u_final_ = []
        m_final = []
        l_final = []
        for level in range(nlevels):
            u.append(graph.get_tensor_by_name("u{}:0".format(level)))
            u_.append(graph.get_tensor_by_name("u_{}:0".format(level)))
            m.append(graph.get_tensor_by_name("m{}:0".format(level)))
            l.append(graph.get_tensor_by_name("l{}:0".format(level)))
            u_final.append(graph.get_tensor_by_name("u_final{}:0".format(level)))
            u_final_.append(graph.get_tensor_by_name("u_final_{}:0".format(level)))
            m_final.append(graph.get_tensor_by_name("m_final{}:0".format(level)))
            l_final.append(graph.get_tensor_by_name("l_final{}:0".format(level)))

        print("Running optimization")

        for step in range(niter_steps):
            print("  Step", step + 1, "/", niter_steps)
            feed_dict = {}
            feed_dict[d] = datacost[None]
            for level in range(nlevels):
                feed_dict[u[level]] = u_init[level]
                feed_dict[u_[level]] = u_init_[level]
                feed_dict[m[level]] = m_init[level]
                feed_dict[l[level]] = l_init[level]

            probs, u_init, u_init_, m_init, l_init = sess.run(
                [p, u_final, u_final_, m_final, l_final], feed_dict=feed_dict)

    return probs[0]



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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--datacost_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--label_map_path")

    parser.add_argument("--nlevels", type=int, default=3)
    parser.add_argument("--nclasses", type=int, required=True)
    parser.add_argument("--nrows", type=int, default=24)
    parser.add_argument("--ncols", type=int, default=24)
    parser.add_argument("--nslices", type=int, default=24)

    parser.add_argument("--niter", type=int, default=10)
    parser.add_argument("--sig", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--nepochs", type=int, default=1000)
    parser.add_argument("--epoch_npasses", type=int, default=1)
    parser.add_argument("--val_nbatches", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--softmax_scale", type=float, default=10)

    parser.add_argument("--niter_steps", type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(0)
    tf.set_random_seed(0)

    tf.logging.set_verbosity(tf.logging.INFO)

    params = {
        "nepochs": args.nepochs,
        "epoch_npasses": args.epoch_npasses,
        "val_nbatches": args.val_nbatches,
        "batch_size": args.batch_size,
        "nlevels": args.nlevels,
        "nclasses": args.nclasses,
        "nrows": args.nrows,
        "ncols": args.ncols,
        "nslices": args.nslices,
        "niter": args.niter,
        "niter_steps": args.niter_steps,
        "sig": args.sig,
        "tau": args.tau,
        "lam": args.lam,
        "learning_rate": args.learning_rate,
        "softmax_scale": args.softmax_scale,
    }

    probs = eval_model(args.checkpoint_path, args.datacost_path, params)

    mkdir_if_not_exists(args.output_path)

    np.savez_compressed(os.path.join(args.output_path, "probs.npz"),
                        probs=probs)

    if args.label_map_path:
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

    for label in range(probs.shape[-1]):
        if args.label_map_path:
            path = os.path.join(args.output_path,
                                "{}-{}.ply".format(label, label_names[label]))
            color = label_colors[label]
        else:
            path = os.path.join(args.output_path, "{}.ply".format(label))
            color = None

        extract_mesh_marching_cubes(path, probs[..., label], color=color)


if __name__ == "__main__":
    main()
