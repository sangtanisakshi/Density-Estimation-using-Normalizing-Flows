import jax
import flax
import tensorflow as tf
import numpy as np
import jax.numpy as jnp
import model as mo
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
    
def get_neighbours(i):

    mat = np.arange(0,16).reshape(4,4)
    loc = np.where(mat == i)
        
    ret = []
        
    if loc[0] != 0:
        ret.append(mat[loc[0] - 1,loc[1]])
        
    if loc[0] != 3:
        ret.append(mat[loc[0] + 1, loc[1]])
            
    if loc[1] != 0:
        ret.append(mat[loc[0], loc[1] - 1])
        
    if loc[1] != 3:
        ret.append(mat[loc[0], loc[1] + 1])
            
    return np.array(ret).squeeze()

def get_patch_size(inputs, dilation):

# for a 32x32 image 1. 8x8 patches - hp=wp=8 for no dilation and for dilation hp=wp=4; Gives 16 patches
# for a 64x64 image 1. 8x8 patches - hp=wp=8 for no dilation and for dilation hp=wp=8; Gives 64 patches
# for a 64x64 image 2. 16x16 patches - hp=wp=16 for no dilation and for dilation hp=wp=4; Gives 16 patches
    img_size = (inputs.shape)[1]
    if dilation:
        patch_size = 4
    else:
        if img_size==32:
            patch_size = 8
        elif img_size==64:
            patch_size = 16
    return patch_size
    
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
            
            
def plot_image_grid(y, title=None, display=True, save_path=None, figsize=(10, 10),recon=False):
    """Plot and optionally save an image grid with matplotlib"""
    fig = plt.figure(figsize=figsize)
    if recon == True:
        num_rows = int(np.floor((np.sqrt(y.shape[0]))/4))
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
        
def sanity_check(random_key):
    # Input
    x_1 = jax.random.normal(random_key, (32, 32, 32, 3))
    K, L = 48, 1
    model = mo.GLOW(K=K, L=L, nn_width=512, key=random_key, learn_top_prior=True)
    init_variables = model.init(random_key, x_1)

    # Forward call
    _, z, logdet, priors = model.apply(init_variables, x_1)

    # Check output shape
    expected_h = x_1.shape[1] 
    expected_c = x_1.shape[-1]
    print("  \033[92m✓\033[0m" if z[-1].shape[1] == expected_h and z[-1].shape[-1] == expected_c 
          else "  \033[91mx\033[0m",
          "Forward pass output shape is", z[-1].shape)

    # Check sizes of the intermediate latent
    correct_latent_shapes = True
    correct_prior_shapes = True
    for i, (zi, priori) in enumerate(zip(z, priors)):
        expected_h = x_1.shape[1]
        expected_c = x_1.shape[-1]
        if zi.shape[1] != expected_h or zi.shape[-1] != expected_c:
            correct_latent_shapes = False
        if priori.shape[1] != expected_h or priori.shape[-1] != 2 * expected_c:
            correct_prior_shapes = False
    print("  \033[92m✓\033[0m" if correct_latent_shapes else "  \033[91mx\033[0m",
          "Check intermediate latents shape")
    print("  \033[92m✓\033[0m" if correct_latent_shapes else "  \033[91mx\033[0m",
          "Check intermediate priors shape")

    # Reverse the network without sampling
    x_3, *_ = model.apply(init_variables, z[-1], z=z, reverse=True)

    print("  \033[92m✓\033[0m" if np.array_equal(x_1.shape, x_3.shape) else "  \033[91mx\033[0m", 
          "Reverse pass output shape = Original shape =", x_1.shape)
    diff = jnp.mean(jnp.abs(x_1 - x_3))
    print("  \033[92m✓\033[0m" if diff < 1e-4 else "  \033[91mx\033[0m", 
          f"Diff between x and Glow_r o Glow (x) = {diff:.3e}")
    