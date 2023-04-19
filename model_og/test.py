## Einops
import jax
import jax.numpy as jnp
import flax
from model import GLOW
import numpy as np

# import glob
# filelist = glob.glob('../../lfw/lfw-deepfunneled/lfw-deepfunneled/*/*.jpg')

# images = np.array([np.array(Image.open(fname)) for fname in filelist])

# images.dump('images.npy')

model_path = '../weights/glow_trial_16/model_epoch=001.weights'
model = GLOW(K=48, L=3, nn_width=512, learn_top_prior=True)

with open(model_path, 'rb') as f:
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((9, 64, 64, 3)))
    params = flax.serialization.from_bytes(params, f.read())

param_count = sum(x.size for x in jax.tree_leaves(params))
print(param_count)
