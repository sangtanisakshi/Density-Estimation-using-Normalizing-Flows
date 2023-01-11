## Einops
import numpy as np

from PIL.Image import fromarray
from IPython import get_ipython

import einops
import numpy as np
from PIL import Image

import glob
filelist = glob.glob('../lfw/lfw-deepfunneled/lfw-deepfunneled/*/*.jpg')

images = np.array([np.array(Image.open(fname)) for fname in filelist])

images.dump('images.npy')