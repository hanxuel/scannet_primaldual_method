import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os.path

from PIL import Image
from scipy import ndimage

from skimage import io
from skimage.draw import draw
from skimage._shared.utils import warn
from skimage.util import random_noise
from numpy import linalg as linalg

from nibabel import eulerangles


def _generate_circle_mask(center_y, center_x, radius):
    """Generate a mask for a filled circle shape.

    The radius of the circle is generated randomly.

    Parameters
    ----------
    center : Point
        The row and column of the top left corner of the rectangle.
    image : ImageShape
        The size of the image into which this shape will be fit.
    shape : ShapeProperties
        The minimum and maximum size and color of the shape to fit.
    random : np.random.RandomState
        The random state to use for random sampling.

    Raises
    ------
    ArithmeticError
        When a shape cannot be fit into the image with the given starting
        coordinates (x_0, y_0). This usually means the image dimensions are too
        small or shape dimensions too large.

    Returns
    -------
    label : Label
        A tuple specifying the category of the shape, as well as its x1, x2, y1
        and y2 bounding box coordinates.
    indices: 2-D array
        A mask of indices that the shape fills.
    """

    circle = draw.circle(center_y, center_x, radius)

    return circle



def conv3d(value, weights, padding="SYMMETRIC"):
    """ 2d convolution - modularize code """
    return tf.nn.conv3d(value, weights, strides=[1, 1, 1, 1, 1], padding="SAME")


def weight_variable_const(name, value):
    """ Create and initialize weights variable """
    return tf.get_variable(
        name, dtype=tf.float32,
        initializer=value,
    )


def weight_grad_variable(name, shape):
    """ Create and initialize weights variable """

    nclasses = shape[2]

    weight_value = np.zeros([2, 2, nclasses, nclasses, 2], dtype=np.float32)
    for c in range(nclasses):
        weight_value[0, 0, c, c, 0] = -1
        weight_value[0, 1, c, c, 0] = 1

        weight_value[0, 0, c, c, 1] = -1
        weight_value[1, 0, c, c, 1] = 1

    weight_value = np.reshape(weight_value, [2, 2, nclasses, nclasses * 2])

    return tf.get_variable(name, dtype=tf.float32, initializer=weight_value)



def main ():
    """ Main function """

    # List of classes
    classes = [
        'sky',
        'building',
        'ground',
        'vegetation',
        'clutter'
    ]

    # Load the weights
    weights_file = "features.npz"
    # weights_file = "weights_shape_33.npz"
    weights = np.load(weights_file)
    weights = weights['weights0/w1']
    # weights = weights.reshape(list(weights.shape[:-1]) +
    #                           [weights.shape[-1] // 3, 3])
    # weights = np.concatenate([weights[:, 0, :, :, :, 0][..., None],
    #                           weights[:, 0, :, :, :, 2][..., None]], axis=-1)
    # print(weights.shape)
    # weights = weights.reshape(list(weights.shape[:-2]) +
    #                           [weights.shape[-2] * 2])
    nclasses = weights.shape[-2]

    weights_tf = weight_variable_const("w", weights)

    # Sample a line along the unit circle
    nsamples = 36
    x_angles = np.linspace(0, 2*np.pi, nsamples)
    z_angles = np.linspace(0, 2*np.pi, nsamples)

    # Create an image with line
    image = np.zeros([nsamples,2,2,2]  , dtype=np.float32)

    # Save images
    img_fold = "Img_temp"
    if not os.path.exists(img_fold):
        os.mkdir(img_fold)

    # Save figures
    fig_fold = "Figures"
    if not os.path.exists(fig_fold):
        os.mkdir(fig_fold)

    # Draw a line
    for samp_x in range(nsamples):
        normal = np.dot(eulerangles.euler2mat(0, 0, x_angles[samp_x]), [0, 1, 0])

        # Create the image
        for i in [-1,1]:
            for j in [-1,1]:
                for k in [-1,1]:
                    vox  = np.array([i, j, k])

                    dist = np.dot(normal, vox) / linalg.norm(normal)
                    val  = max(0, 1-abs(dist))

                    print(i, j, k, dist)

                    if dist > 0:
                        image[samp_x, 0 if i == -1 else 1,
                                      0 if j == -1 else 1,
                                      0 if k == -1 else 1] = max(val, min(1, dist))
                    else:
                        image[samp_x, 0 if i == -1 else 1,
                                      0 if j == -1 else 1,
                                      0 if k == -1 else 1] = val

        # io.imsave("img-{}.png".format(samp_x), (255 * image[samp_x,0]).astype(np.uint8))
        # io.imsave("img-1-{}.png".format(samp_x), (255 * image[samp_x,1]).astype(np.uint8))

        # return

    # Apply the convolution
    probs_tf = tf.placeholder(dtype=tf.float32, shape=[None, 2, 2, 2, nclasses])
    res_tf = conv3d(probs_tf, weights_tf)

    # Apply to many things
    for c in range(nclasses):

        if not os.path.exists('{}/{}'.format(fig_fold, classes[c])):
            os.mkdir('{}/{}'.format(fig_fold, classes[c]))

        print('{}/{}'.format(fig_fold, classes[c]))

        for d in range(nclasses):

            # Prepare the probabilites
            probs = np.zeros([nsamples, 2, 2, 2, nclasses], dtype=np.float32)
            if c == d:
                probs[:, :, :, :, c] = np.ones([nsamples, 2, 2, 2], dtype=np.float32)
            else:
                probs[:, :, :, :, c] = image
                probs[:, :, :, :, d] = 1 - image

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                res = sess.run([
                    res_tf
                    ],
                    feed_dict={
                        probs_tf: probs
                    }
                )

                res = res[0]
                res_x = np.abs(res[:,0,0,0,:nclasses])
                res_y = np.abs(res[:,0,0,0,nclasses:2*nclasses])
                res_z = np.abs(res[:,0,0,0,2*nclasses:])

                res_vis = res_x + res_y + res_z

            plt.style.use("ggplot")

            # Save results
            plt.clf()
            res_vis = np.sum(res_vis, axis=1)
            fig = plt.figure(1)
            ax = plt.subplot(111, projection='polar')
            ax.plot(x_angles, res_vis)

            plt.tight_layout()

            plt.savefig('{}/{}/{}_to_{}.png'.format(fig_fold, classes[c], classes[c], classes[d]))


if __name__ == '__main__':
    main()
