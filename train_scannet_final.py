import os
import glob
import argparse
import numpy as np
import tensorflow as tf
import multiprocessing
import joblib
import collections
ProcessedItem = collections.namedtuple(
    'ProcessedItem', ['groundtruth', 'datacost'])

EPSILON = 10e-8


def mkdir_if_not_exists(path):
    assert os.path.exists(os.path.dirname(path.rstrip("/")))
    if not os.path.exists(path):
        os.makedirs(path)


def conv_weight_variable(name, shape, stddev=1.0):
    #initializer = tf.truncated_normal_initializer(stddev=stddev)
    return tf.get_variable(name, shape, dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=stddev))


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

def is_sample_valid(groundtruth, occupancy_threshold):
        """ Tests whether the given voxelgrid represents a valid training sample, based on its percent occupancy and percent annotated

        Args:
        voxelgrid: np.array, a 3D grid
        occupancy_threshold: float
        annotation_threshold: float

        Returns: bool

        """

        num_unoccupied_voxels = len(np.where(groundtruth[:,:,:,-1]==1)[0])
        num_voxels = np.product(groundtruth.shape[:3])
        percent_unoccupied = num_unoccupied_voxels / num_voxels
        percent_occupied = 1 - percent_unoccupied

        if percent_occupied < occupancy_threshold:
            return False
        return True
    
def get_list_dir(scene_path):
    working_dir = os.getcwd()

    os.chdir(scene_path)
    file_list = [] 

    for file in glob.glob("*/"):
        file_list.append(file)

    os.chdir(working_dir)

    return file_list


def get_list_files(scene_path, ext):
    working_dir = os.getcwd()

    os.chdir(scene_path)
    file_list = [] 

    for file in glob.glob("*{}".format(ext)):
        file_list.append(file)

    file_list = sorted(file_list, key=lambda name: int(name[:6]))

    os.chdir(working_dir)

    return file_list

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

'''
def categorical_crossentropy(y_true, y_pred, params):
    nclasses = y_true.shape[-1]

    with tf.name_scope("categorical_cross_entropy"):
        labels_true = tf.argmax(y_true, axis=-1)
        freespace_label = y_true.shape[-1] - 1
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

        # Final per voxel loss function weights.
        weights = tf.maximum(EPSILON, unkown_weights * unobserved_weights)
        # Compute weighted cross entropy.
        cross_entropy = -tf.reduce_sum(weights * y_true * tf.log(y_pred)) / \
                         tf.reduce_sum(weights)

    return cross_entropy
'''
def categorical_crossentropy(y_true, y_pred, params):
    nclasses = y_true.shape[-1]

    with tf.name_scope("categorical_cross_entropy"):
        y_true = tf.clip_by_value(y_true, EPSILON, 1.0 - EPSILON)
        y_pred = tf.clip_by_value(y_pred, EPSILON, 1.0 - EPSILON)

        # Measure how close we are to unknown class.
        # unkown_weights = 1 - y_true[..., -2][..., None]
        freespace_label = y_true.shape[-1] - 1
        freespace_mask = tf.equal(tf.argmax(y_true, axis=-1), freespace_label)
        known_mask     = ~tf.equal(tf.argmax(y_true, axis=-1), freespace_label-1)
        occupied_mask  = tf.logical_and(~freespace_mask, known_mask)

        # Compute weighted cross entropy.
        cross_entropy = y_true * tf.log(y_pred)
        print(tf.reduce_sum(tf.cast(freespace_mask, tf.float32)))  
        # Compute weighted cross entropy.
        freespace_cross_entropy = \
            -tf.reduce_sum(tf.boolean_mask(cross_entropy, freespace_mask)) / \
             (tf.reduce_sum(tf.cast(freespace_mask, tf.float32)) + EPSILON)

        occupied_cross_entropy = \
            -tf.reduce_sum(tf.boolean_mask(cross_entropy, occupied_mask)) / \
             (tf.reduce_sum(tf.cast(occupied_mask, tf.float32)) + EPSILON)

        cross_entropy = freespace_cross_entropy + params["loss_weight"]*occupied_cross_entropy

    return cross_entropy


def build_model(params):
    batch_size = params["batch_size"]
    nlevels = params["nlevels"]
    nrows = params["nrows"]
    ncols = params["ncols"]
    nslices = params["nslices"]
    nclasses = params["nclasses"]
    softmax_scale = params["softmax_scale"]
    #with tf.variable_scope("reconstruction"):
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

    for iter in range(10):
        u_loop, u_loop_, m_loop, l_loop = primal_dual(
                u_loop, u_loop_, m_loop, l_loop, d_encoded, sig, tau, iter)
    with tf.device('/gpu:' + str(1)):
        for iter in range(10,25,1):
            u_loop, u_loop_, m_loop, l_loop = primal_dual(
                    u_loop, u_loop_, m_loop, l_loop, d_encoded, sig, tau, iter)
    with tf.device('/gpu:' + str(2)):
        for iter in range(25,40,1):
            u_loop, u_loop_, m_loop, l_loop = primal_dual(
                    u_loop, u_loop_, m_loop, l_loop, d_encoded, sig, tau, iter)
    with tf.device('/gpu:' + str(3)):
        for iter in range(40,50,1):
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

def preprocess_dataset_item(scene_path,scene_id,nclasses):
    datacost_path = os.path.join(scene_path, scene_id,"groundtruth_datacost.npz")
    groundtruth_path = os.path.join(scene_path, scene_id,
                                    "groundtruth_model","groundtruth.npz")
    
    if not os.path.exists(datacost_path) or not os.path.exists(groundtruth_path):
        print("  Warning: datacost or groundtruth does not exist")
        return ProcessedItem([], [])
    
    datacost_data = np.load(datacost_path)
    datacost=np.zeros(datacost_data["shape"],dtype=np.float32)
    datacost[datacost_data["idxs"][0],datacost_data["idxs"][1],datacost_data["idxs"][2],datacost_data["idxs"][3]]=datacost_data["values"]
            #datacost = datacost_data["volume"]

    groundtruth_data = np.load(groundtruth_path)
    groundtruth=np.zeros(groundtruth_data["shape"],dtype=np.float32)
    groundtruth[groundtruth_data["idxs"][0],groundtruth_data["idxs"][1],groundtruth_data["idxs"][2],groundtruth_data["idxs"][3]]=groundtruth_data["values"]

        # Make sure the data is compatible with the parameters.
    assert datacost.shape[3] == nclasses
    assert datacost.shape == groundtruth.shape
    print("extracted data from {}".format(scene_id))
    return ProcessedItem(groundtruth,datacost)


def build_data_generator(data_path,params):
    epoch_npasses = params["epoch_npasses"]
    batch_size = params["batch_size"]
    nrows = params["nrows"]
    ncols = params["ncols"]
    nslices = params["nslices"]
    nclasses = params["nclasses"]

    #scene_list = []
    scene_list = get_list_dir(data_path)

    idxs = np.arange(len(scene_list))
    print("-------------------------number of scene is{}--------------------------------".format(len(scene_list)))

    batch_datacost = np.empty(
        (batch_size, nrows, ncols, nslices, nclasses), dtype=np.float32)
    batch_groundtruth = np.empty(
        (batch_size, nrows, ncols, nslices, nclasses), dtype=np.float32)

    npasses = 0

    while True:
        # Shuffle all data samples.
        np.random.shuffle(idxs)

        # One epoch iterates once over all scenes.
        for batch_start_idx in range(0, len(idxs), batch_size):
            # Determine the random scenes for current batch.
            batch_end_idx = min(batch_start_idx + batch_size, len(idxs))
            batch_idxs = idxs[batch_start_idx:batch_end_idx]

            # By default, set all voxels to unobserved.
            batch_datacost[:] = 0
            batch_groundtruth[:] = 1.0 / nclasses

            # Prepare data for random scenes in current batch.
            for i, idx in enumerate(batch_idxs):
                datacost_path = os.path.join(data_path, scene_list[idx],"groundtruth_datacost.npz")
                groundtruth_path = os.path.join(data_path, scene_list[idx],"groundtruth.npz")
                datacost_data = np.load(datacost_path)
                datacost=np.zeros(datacost_data["shape"],dtype=np.float32)
                datacost[datacost_data["idxs"][0],datacost_data["idxs"][1],datacost_data["idxs"][2],datacost_data["idxs"][3]]=datacost_data["values"]
            #datacost = datacost_data["volume"]

                groundtruth_data = np.load(groundtruth_path)
                groundtruth=np.zeros(groundtruth_data["shape"],dtype=np.float32)
                groundtruth[groundtruth_data["idxs"][0],groundtruth_data["idxs"][1],groundtruth_data["idxs"][2],groundtruth_data["idxs"][3]]=groundtruth_data["values"]
                assert datacost.shape[3] == nclasses
                assert datacost.shape == groundtruth.shape
                m=False
                
                while m==False:
                    # Determine a random crop of the input data.
                    row_start = np.random.randint(
                            0, max(datacost.shape[0] - nrows, 0) + 1)
                    col_start = np.random.randint(
                            0, max(datacost.shape[1] - ncols, 0) + 1)
                    slice_start = np.random.randint(
                            0, max(datacost.shape[2] - nslices, 0) + 1)
                    row_end = min(row_start + nrows, datacost.shape[0])
                    col_end = min(col_start + ncols, datacost.shape[1])
                    slice_end = min(slice_start + nslices, datacost.shape[2])
                
                    if is_sample_valid(groundtruth=groundtruth[row_start:row_end,col_start:col_end,slice_start:slice_end],\
                                       occupancy_threshold=0.001):
                        # Copy the random crop of the data cost.
                        batch_datacost[i,
                                       :row_end-row_start,
                                       :col_end-col_start,
                                       :slice_end-slice_start] = \
                                       datacost[row_start:row_end,
                                                col_start:col_end,
                                                slice_start:slice_end]
                        # Copy the random crop of the groundtruth.
                        batch_groundtruth[i,
                                          :row_end-row_start,
                                          :col_end-col_start,
                                          :slice_end-slice_start] = \
                                          groundtruth[row_start:row_end,
                                                      col_start:col_end,
                                                      slice_start:slice_end]
                        m=True
                # Randomly rotate around z-axis.
                num_rot90 = np.random.randint(4)
                if num_rot90 > 0:
                    batch_datacost[i] = np.rot90(batch_datacost[i],
                                                 k=num_rot90,
                                                 axes=(0, 1))
                    batch_groundtruth[i] = np.rot90(batch_groundtruth[i],
                                                    k=num_rot90,
                                                    axes=(0, 1))

                # Randomly flip along x and y axis.
                flip_axis = np.random.randint(3)
                if flip_axis == 0 or flip_axis == 1:
                    batch_datacost[i] = np.flip(batch_datacost[i],
                                                axis=flip_axis)
                    batch_groundtruth[i] = np.flip(batch_groundtruth[i],
                                                   axis=flip_axis)

            yield (batch_datacost[:len(batch_idxs)],
                   batch_groundtruth[:len(batch_idxs)])

        npasses += 1

        if epoch_npasses > 0 and npasses >= epoch_npasses:
            npasses = 0
            yield   
def get_learning_rate(batch, batch_size, init_learning_rate, decay_rate, decay_steps, min_learning_rate=0.00001):
    """ Get exponentially decaying learning rate clipped at min_learning_rate.

    Args:
    batch: int
    batch_size: int
    init_learning_rate: float
    decay_rate: float
    decay_steps: int
    min_learning_rate: float

    Returns: float

    """
    learning_rate = tf.train.exponential_decay(
        init_learning_rate, batch, decay_steps, decay_rate, staircase=True)
    return tf.maximum(learning_rate, min_learning_rate)
def train_model(data_path,val_path, model_path, params):
    log_path = os.path.join(model_path, "logs")
    checkpoint_path = os.path.join(model_path, "checkpoint")

    mkdir_if_not_exists(log_path)
    mkdir_if_not_exists(checkpoint_path)

    batch_size = params["batch_size"]
    nlevels = params["nlevels"]
    nrows = params["nrows"]
    ncols = params["ncols"]
    nslices = params["nslices"]
    nclasses = params["nclasses"]
    val_nbatches = params["val_nbatches"]

    train_data_generator = \
        build_data_generator(data_path, params)

    val_params = dict(params)
    val_params["epoch_npasses"] = -1
    val_data_generator = \
        build_data_generator(val_path, val_params)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True

    sess = tf.Session(config=tf_config)

    with sess.as_default():
        probs, datacost, u, u_, m, l = build_model(params)
     
        batch_i = tf.Variable(0, name='batch_i')
        for variable in tf.trainable_variables():
            #if not variable.name.endswith('weights:0'):
            #    continue
            print(variable.name + ' - ' + str(variable.get_shape()) + ' - ' + str(np.prod(variable.get_shape().as_list())))
        groundtruth = tf.placeholder(tf.float32, probs[0].shape, name="groundtruth")
    
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
            u_init.append(np.empty([batch_size, nrows_level, ncols_level,
                                    nslices_level, nclasses],
                                   dtype=np.float32))
            u_init_.append(np.empty([batch_size, nrows_level, ncols_level,
                                     nslices_level, nclasses],
                                    dtype=np.float32))
            m_init.append(np.empty([batch_size, nrows_level, ncols_level,
                                    nslices_level, 3 * nclasses],
                                   dtype=np.float32))
            l_init.append(np.empty([batch_size, nrows_level, ncols_level,
                                    nslices_level],
                                   dtype=np.float32))
    
        loss_op = categorical_crossentropy(groundtruth, probs[0], params)
        freespace_accuracy_op, occupied_accuracy_op, semantic_accuracy_op = \
            classification_accuracy(groundtruth, probs[0])
            
        learning_rate_op=get_learning_rate(
                batch_i, params['batch_size'], params['initial_learning_rate'], params['decay_rate'], params['decay_step'])
        tf.summary.scalar('learning_rate', learning_rate_op)
    
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_op)
        train_op = optimizer.minimize(loss_op,global_step=batch_i)
    
        train_loss_summary = \
            tf.placeholder(tf.float32, name="train_loss_summary")
        train_freespace_accuracy_summary = \
            tf.placeholder(tf.float32, name="train_freespace_accuracy_summary")
        train_occupied_accuracy_summary = \
            tf.placeholder(tf.float32, name="train_occupied_accuracy_summary")
        train_semantic_accuracy_summary = \
            tf.placeholder(tf.float32, name="train_semantic_accuracy_summary")
    
        tf.summary.scalar("train_loss",
                          train_loss_summary)
        tf.summary.scalar("train_freespace_accuracy",
                          train_freespace_accuracy_summary)
        tf.summary.scalar("train_occupied_accuracy",
                          train_occupied_accuracy_summary)
        tf.summary.scalar("train_semantic_accuracy",
                          train_semantic_accuracy_summary)
    
        val_loss_summary = \
            tf.placeholder(tf.float32, name="val_loss_summary")
        val_freespace_accuracy_summary = \
            tf.placeholder(tf.float32, name="val_freespace_accuracy_summary")
        val_occupied_accuracy_summary = \
            tf.placeholder(tf.float32, name="val_occupied_accuracy_summary")
        val_semantic_accuracy_summary = \
            tf.placeholder(tf.float32, name="val_semantic_accuracy_summary")
    
        tf.summary.scalar("val_loss",
                          val_loss_summary)
        tf.summary.scalar("val_freespace_accuracy",
                          val_freespace_accuracy_summary)
        tf.summary.scalar("val_occupied_accuracy",
                          val_occupied_accuracy_summary)
        tf.summary.scalar("val_semantic_accuracy",
                          val_semantic_accuracy_summary)
    
        summary_op = tf.summary.merge_all()
    
        model_saver = tf.train.Saver(save_relative_paths=True)
        train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2,
                                     save_relative_paths=True, pad_step_number=True)
        log_writer = tf.summary.FileWriter(log_path)
        log_writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())

        model_saver.save(sess, os.path.join(checkpoint_path, "initial"),
                         write_meta_graph=True)
        #train_saver.restore(sess, os.path.join(checkpoint_path, "checkpoint-00000022"))
        #print("checkpoint-00000022 has been restored")
       
        for epoch in range(params["nepochs"]):

            train_loss_values = []
            train_freespace_accuracy_values = []
            train_occupied_accuracy_values = []
            train_semantic_accuracy_values = []

            batch = 0
            while True:
                train_data = next(train_data_generator) #for one epoche, if finish, yield none

                # Check if epoch finished.
                if train_data is None:
                    break

                datacost_batch, groundtruth_batch = train_data

                num_batch_samples = datacost_batch.shape[0]

                feed_dict = {}

                feed_dict[datacost] = datacost_batch
                feed_dict[groundtruth] = groundtruth_batch

                for level in range(nlevels):
                    u_init[level][:] = 1.0 / nclasses
                    u_init_[level][:] = 1.0 / nclasses
                    m_init[level][:] = 0.0
                    l_init[level][:] = 0.0
                    feed_dict[u[level]] = u_init[level][:num_batch_samples]
                    feed_dict[u_[level]] = u_init_[level][:num_batch_samples]
                    feed_dict[m[level]] = m_init[level][:num_batch_samples]
                    feed_dict[l[level]] = l_init[level][:num_batch_samples]

                (_,
                 loss,
                 freespace_accuracy,
                 occupied_accuracy,
                 semantic_accuracy) = sess.run(
                    [train_op,
                     loss_op,
                     freespace_accuracy_op,
                     occupied_accuracy_op,
                     semantic_accuracy_op],
                    feed_dict=feed_dict
                )
                print('batch_i:{}'.format(sess.run(batch_i)))
                print('Learning rate: %f' % (sess.run(optimizer._lr_t)))

                train_loss_values.append(loss)
                train_freespace_accuracy_values.append(freespace_accuracy)
                train_occupied_accuracy_values.append(occupied_accuracy)
                train_semantic_accuracy_values.append(semantic_accuracy)

                print("Epoch: {}, "
                      "Batch: {}\n"
                      "  Loss:                  {}\n"
                      "  Free Space Accuracy:   {}\n"
                      "  Occupied Accuracy:     {}\n"
                      "  Semantic Accuracy:     {}".format(
                      epoch + 1,
                      batch + 1,
                      loss,
                      freespace_accuracy,
                      occupied_accuracy,
                      semantic_accuracy))

                batch += 1
                

            train_loss_value = \
                np.nanmean(train_loss_values)
            train_freespace_accuracy_value = \
                np.nanmean(train_freespace_accuracy_values)
            train_occupied_accuracy_value = \
                np.nanmean(train_occupied_accuracy_values)
            train_semantic_accuracy_value = \
                np.nanmean(train_semantic_accuracy_values)

            val_loss_values = []
            val_freespace_accuracy_values = []
            val_occupied_accuracy_values = []
            val_semantic_accuracy_values = []

            for _ in range(val_nbatches):
                datacost_batch, groundtruth_batch = next(val_data_generator)

                num_batch_samples = datacost_batch.shape[0]

                feed_dict = {}

                feed_dict[datacost] = datacost_batch
                feed_dict[groundtruth] = groundtruth_batch

                for level in range(nlevels):
                    u_init[level][:] = 1.0 / nclasses
                    u_init_[level][:] = 1.0 / nclasses
                    m_init[level][:] = 0.0
                    l_init[level][:] = 0.0
                    feed_dict[u[level]] = u_init[level][:num_batch_samples]
                    feed_dict[u_[level]] = u_init_[level][:num_batch_samples]
                    feed_dict[m[level]] = m_init[level][:num_batch_samples]
                    feed_dict[l[level]] = l_init[level][:num_batch_samples]

                (loss,
                 freespace_accuracy,
                 occupied_accuracy,
                 semantic_accuracy) = sess.run(
                    [loss_op,
                     freespace_accuracy_op,
                     occupied_accuracy_op,
                     semantic_accuracy_op],
                    feed_dict=feed_dict
                )

                val_loss_values.append(loss)
                val_freespace_accuracy_values.append(freespace_accuracy)
                val_occupied_accuracy_values.append(occupied_accuracy)
                val_semantic_accuracy_values.append(semantic_accuracy)
            val_loss_value = \
                np.nanmean(val_loss_values)
            val_freespace_accuracy_value = \
                np.nanmean(val_freespace_accuracy_values)
            val_occupied_accuracy_value = \
                np.nanmean(val_occupied_accuracy_values)
            val_semantic_accuracy_value = \
                np.nanmean(val_semantic_accuracy_values)

            print()
            print(78 * "#")
            print()

            print("Validation\n"
                  "  Loss:                  {}\n"
                  "  Free Space Accuracy:   {}\n"
                  "  Occupied Accuracy:     {}\n"
                  "  Semantic Accuracy:     {}".format(
                  val_loss_value,
                  val_freespace_accuracy_value,
                  val_occupied_accuracy_value,
                  val_semantic_accuracy_value))

            print()
            print(78 * "#")
            print(78 * "#")
            print()

            summary = sess.run(
                summary_op,
                feed_dict={
                    train_loss_summary:
                        train_loss_value,
                    train_freespace_accuracy_summary:
                        train_freespace_accuracy_value,
                    train_occupied_accuracy_summary:
                        train_occupied_accuracy_value,
                    train_semantic_accuracy_summary:
                        train_semantic_accuracy_value,
                    val_loss_summary:
                        val_loss_value,
                    val_freespace_accuracy_summary:
                        val_freespace_accuracy_value,
                    val_occupied_accuracy_summary:
                        val_occupied_accuracy_value,
                    val_semantic_accuracy_summary:
                        val_semantic_accuracy_value,
                }
            )

            log_writer.add_summary(summary, epoch)
            train_saver.save(sess, os.path.join(checkpoint_path, "checkpoint"),
                             global_step=epoch, write_meta_graph=False)

        model_saver.save(sess, os.path.join(checkpoint_path, "final"),
                         write_meta_graph=True)
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scene_train_path", default="/cluster/scratch/haliang/SUNCG/train_ceil5_wall_5")
    parser.add_argument("--scene_val_path", default="/cluster/scratch/haliang/SUNCG/val_ceil5_wall_5")
    #parser.add_argument("--scene_train_list_path", required=True)
    #parser.add_argument("--scene_val_list_path", required=True)
    parser.add_argument("--model_path", required=True)

    parser.add_argument("--nclasses", type=int, default=38)

    parser.add_argument("--nlevels", type=int, default=3)
    parser.add_argument("--nrows", type=int, default=24)
    parser.add_argument("--ncols", type=int, default=24)
    parser.add_argument("--nslices", type=int, default=24)

    parser.add_argument("--niter", type=int, default=30)
    parser.add_argument("--sig", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--nepochs", type=int, default=1000)
    parser.add_argument("--epoch_npasses", type=int, default=1)
    parser.add_argument("--val_nbatches", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--softmax_scale", type=float, default=10)
    parser.add_argument("--initial_learning_rate", type=float, default=0.0001)
    parser.add_argument("--decay_rate", type=float, default=0.99)
    parser.add_argument("--decay_step", type=int, default=90) 
    parser.add_argument("--loss_weight", type=float, default=5)
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
        "sig": args.sig,
        "tau": args.tau,
        "lam": args.lam,
        "learning_rate": args.learning_rate,
        "softmax_scale": args.softmax_scale,
        "initial_learning_rate":args.initial_learning_rate,
        "decay_rate":args.decay_rate,
        "decay_step":args.decay_step,
        "loss_weight":args.loss_weight
    }

    train_model(args.scene_train_path, args.scene_val_path, args.model_path, params)


if __name__ == "__main__":
    main()