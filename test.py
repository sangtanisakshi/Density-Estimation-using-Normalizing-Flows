## Einops
import numpy as np

from PIL.Image import fromarray
from IPython import get_ipython

import einops
import numpy as np
from PIL import Image

import glob
filelist = glob.glob('../../lfw/lfw-deepfunneled/lfw-deepfunneled/*/*.jpg')

images = np.array([np.array(Image.open(fname)) for fname in filelist])

images.dump('images.npy')

# model_path = 'weights/model_epoch=070.weights'
# model = GLOW(K=32, L=3, nn_width=512, learn_top_prior=True)

# with open(model_path, 'rb') as f:
#     params = model.init(PRNGKey(0), jnp.zeros((9, 64, 64, 3)))
#     params = flax.serialization.from_bytes(params, f.read())

# param_count = sum(x.size for x in jax.tree_leaves(params))
# print(param_count)
