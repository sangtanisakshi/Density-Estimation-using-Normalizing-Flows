import jax
import flax
import tensorflow as tf

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def map_fn(image_path, num_bits=5, size=256, training=True):
    """Read image file, quantize and map to [-0.5, 0.5] range.
    If num_bits = 8, there is no quantization effect."""
    image = tf.io.decode_jpeg(tf.io.read_file(image_path))
    # Resize input image
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (size, size))
    image = tf.clip_by_value(image, 0., 255.)
    # Discretize to the given number of bits
    if num_bits < 8:
        image = tf.floor(image / 2 ** (8 - num_bits))
    # Send to [-1, 1]
    num_bins = 2 ** num_bits
    image = image / num_bins - 0.5
    if training:
        image = image + tf.random.uniform(tf.shape(image), 0, 1. / num_bins)
    return image

# Utils to display Jax model in a similar way as flax summary
def get_params_size(v, s=0):
    """Get cumulative size of parameters contained in a FrozenDict"""
    if isinstance(v, flax.core.FrozenDict):
        return s + sum(get_params_size(x)  for x in v.values())
    else:
        return s + v.size

def summarize_jax_model(variables, 
                        max_depth=1, 
                        depth=0,
                        prefix='',
                        col1_size=60,
                        col2_size=30):
    """Print summary of parameters + size contained in a jax model"""
    if depth == 0:
        print('-' * (col1_size + col2_size))
        print("Layer name" + ' ' * (col1_size - 10) + 'Param #')
        print('=' * (col1_size + col2_size))
    for name, v in variables.items():
        if isinstance(v, flax.core.FrozenDict) and depth < max_depth:
            summarize_jax_model(v, max_depth=max_depth, depth=depth + 1, 
                                prefix=f'{prefix}/{name}')
        else:
            col1 = f'{prefix}/{name}'
            col1 = col1[:col1_size] + ' ' * max(0, col1_size - len(col1))
            print(f'{col1}{get_params_size(v)}')
            print('-' * (col1_size + col2_size))
            
            
def plot_image_grid(y, title=None, display=True, save_path=None, figsize=(10, 10)):
    """Plot and optionally save an image grid with matplotlib"""
    fig = plt.figure(figsize=figsize)
    num_rows = int(np.floor(np.sqrt(y.shape[0])))
    grid = ImageGrid(fig, 111, nrows_ncols=(num_rows, num_rows), axes_pad=0.1)
    for ax in grid: 
        ax.set_axis_off()
    for ax, im in zip(grid, y):
        ax.imshow(im)
    fig.suptitle(title, fontsize=18)
    fig.subplots_adjust(top=0.98)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if display:
        plt.show()
    else:
        plt.close()