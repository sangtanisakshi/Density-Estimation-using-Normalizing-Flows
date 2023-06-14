from jax.random import PRNGKey, normal
import jax.numpy as jnp
import flax
import jax
import model as mo
import argparse
import sample as sam
from utils import plot_image_grid
from functools import partial

parser = argparse.ArgumentParser(description='Sample from pretrained model.')
parser.add_argument('num_samples', type=int, help='number of samples')
parser.add_argument('-t', '--temperature', default=0.7, type=float, help='Temperature')
parser.add_argument('-s', '--seed', type=int, default=42, help='random seed')
parser.add_argument('--model_path', type=str, help='Model path')
args = parser.parse_args()


model = mo.GLOW(K=48, L=1, nn_width=512, learn_top_prior=True, dilation=True, only_neighbours=False)

with open(args.model_path, 'rb') as f:
    params = model.init(PRNGKey(args.seed), jnp.zeros((
        args.num_samples, 64, 64, 3)))
    params = flax.serialization.from_bytes(params, f.read())

fname = (args.model_path).split(".")[0]
    
sam.sample(model, params, shape=(args.num_samples, 8, 8, 3),
       key=PRNGKey(args.seed), 
       sampling_temperature=args.temperature,
       postprocess_fn=partial(sam.postprocess, num_bits=5),
       save_path=f"models/{fname}_sample_seed={args.seed}_t={args.temperature}.png")