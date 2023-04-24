import jax
import flax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import os
import argparse
import glob
import tensorflow as tf
import tensorflow_datasets as tfds

print(jax.devices())
##TODO - remove this because it's if True then flax needs to be 0.6.1 or higher
## possible that it might be making things slower because it only uses numpy array and not jax array
jax.config.update('jax_array', False)
import time
import wandb

from functools import partial
from matplotlib import pyplot as plt

from sample import postprocess
from sample import sample
from model import GLOW
from utils import summarize_jax_model
from utils import plot_image_grid
from utils import map_fn
from PIL import Image

print('Jax version', jax.__version__)
print('Flax version', flax.__version__)

random_key = jax.random.PRNGKey(0)

parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('-name','--wb_name', default="trial-test", type=str, help='WandB Run Name', required=False)
parser.add_argument('-desc', '--wb_desc', type=str, help='WandB Run Description', required=False)
#parser.add_argument('-hpo', '--hpo', default=False, type=bool, help='Hyperparameter Optimization', required=False)
args = parser.parse_args()

@jax.vmap
def get_logpz(z, priors):
    logpz = 0
    for zi, priori in zip(z, priors):
        if priori is None:
            mu = jnp.zeros(zi.shape)
            logsigma = jnp.zeros(zi.shape)
        else:
            mu, logsigma = jnp.split(priori, 2, axis=-1)
        logpz += jnp.sum(- logsigma - 0.5 * jnp.log(2 * jnp.pi) 
                         - 0.5 * (zi - mu) ** 2 / jnp.exp(2 * logsigma))
    return logpz

# Data hyperparameters for 1 GPU training
config_dict = {
    'image_path': "/dhc/home/sakshi.sangtani/lfw/lfw-deepfunneled/lfw-deepfunneled/*",
    'train_split': 0.8,
    'image_size': 32,
    'num_channels': 3,
    'num_bits': 8,
    'batch_size': 256,
    'K': 16,
    'L': 1,
    'nn_width': 512,
    'learn_top_prior': True,
    'sampling_temperature': 0.7,
    'init_lr': 1e-3,
    'num_epochs': 5,
    'num_warmup_epochs': 8, # For learning rate warmup
    'num_sample_epochs': 5, # Fractional epochs for sampling because one epoch is quite long 
    'num_save_epochs': 1,
    'num_samples': 9
}

# Hyperparameter Optimization
#if args.hpo == true:
#   with open('./wandb_sweep.yaml') as file:
#       hpo_config = yaml.load(file, Loader=yaml.FullLoader)

#Initialize WandB
wandb.init(project="research-nf", entity="dhc_research", name= args.wb_name, config=config_dict, notes=args.wb_desc, dir="../")

output_hw = config_dict["image_size"] // 2 ** config_dict["L"]
output_c = config_dict["num_channels"] * 4**config_dict["L"] // 2**(config_dict["L"] - 1)
config_dict["sampling_shape"] = (output_hw, output_hw, output_c)

tf.config.experimental.set_visible_devices([], 'GPU')

def get_train_dataset(img_data,image_size, num_bits, batch_size, skip=None, **kwargs):
    del kwargs
    train_ds = img_data
    if skip is not None:
        train_ds = train_ds.skip(skip)
    train_ds = train_ds.shuffle(buffer_size=200)
    train_ds = train_ds.map(partial(map_fn, size=image_size, num_bits=num_bits, training=True))
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.repeat()
    return iter(tfds.as_numpy(train_ds))


def get_val_dataset(img_data,image_size, num_bits, batch_size, 
                    take=None, repeat=False, **kwargs):
    del kwargs
    val_ds = img_data
    if take is not None:
        val_ds = val_ds.take(take)
    val_ds = val_ds.map(partial(map_fn, size=image_size, num_bits=num_bits, training=False))
    val_ds = val_ds.batch(batch_size)
    if repeat:
        val_ds = val_ds.repeat()
    return iter(tfds.as_numpy(val_ds))

##Make all directories
weights_loc ="../weights/"+args.wb_name+"/"
if not os.path.exists(weights_loc): os.makedirs(weights_loc)
fp = "../samples/"+args.wb_name+"/"
if not os.path.exists(fp): os.makedirs(fp)
results_loc = "../results/"+args.wb_name+"/"
if not os.path.exists(results_loc): os.makedirs(results_loc)

def train_glow(train_ds,
               val_ds=None,
               num_samples=9,
               image_size=256,
               num_channels=3,
               num_bits=8,
               init_lr=1e-3,
               num_epochs=1,
               num_sample_epochs=1,
               num_warmup_epochs=10,
               num_save_epochs=2,
               steps_per_epoch=1,
               K=48,
               L=1,
               nn_width=512,
               sampling_temperature=0.7,
               learn_top_prior=True,
               key=jax.random.PRNGKey(0),
               **kwargs):
    """Simple training loop.
    Args:
        train_ds: Training dataset iterator (e.g. tensorflow dataset)
        val_ds: Validation dataset (optional)
        num_samples: Number of samples to generate at each epoch
        image_size: Input image size
        num_channels: Number of channels in input images
        num_bits: Number of bits for discretization
        init_lr: Initial learning rate (Adam)
        num_epochs: Numer of training epochs
        num_sample_epochs: Visualize sample at this interval
        num_warmup_epochs: Linear warmup of the learning rate to init_lr
        num_save_epochs: save model at this interval
        steps_per_epochs: Number of steps per epochs
        K: Number of flow iterations in the GLOW model
        L: number of scales in the GLOW model
        nn_width: Layer width in the Affine Coupling Layer
        sampling_temperature: Smoothing temperature for sampling from the 
            Gaussian priors (1 = no effect)
        learn_top_prior: Whether to learn the prior for highest latent variable zL.
            Otherwise, assumes standard unit Gaussian prior
        key: Random seed
    """
    del kwargs
    # Init model
    model = GLOW(K=K,
                 L=L, 
                 nn_width=config_dict['nn_width'], 
                 learn_top_prior=learn_top_prior,
                 key=key)
    
    # Init optimizer and learning rate schedule
    params = model.init(random_key, next(train_ds))
    opt = flax.optim.Adam(learning_rate=init_lr, weight_decay=0.0001).init(params)
    ##TODO - check beta1 and beta2 values for Adam 
    # Summarize the final model
    summarize_jax_model(params, max_depth=2)
    
    # Learning rate warmup 
    def lr_warmup(step):
        return init_lr * jnp.minimum(1., step / (num_warmup_epochs * steps_per_epoch + 1e-8))
    
    # Helper functions for training
    bits_per_dims_norm = np.log(2.) * num_channels * image_size**2
    @jax.jit
    def get_logpx(z, logdets, priors):
        logpz = get_logpz(z, priors)
        logpz = jnp.mean(logpz) / bits_per_dims_norm        # bits per dimension normalization
        logdets = jnp.mean(logdets) / bits_per_dims_norm
        logpx = logpz + logdets - num_bits                  # num_bits: dequantization factor
        return logpx, logpz, logdets
        
    # @jax.jit
    # def train_step(params, opt_state, batch):
    #     def loss_fn(params):
    #         _, z, logdets, priors = model.apply(params, batch, reverse=False)
    #         logpx, logpz, logdets = get_logpx(z, logdets, priors)
    #         return - logpx, (logpz, logdets)
    #     logs, grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
    #     updates, opt_state = opt.update(grad, opt_state, params)
    #     params = optax.apply_updates(params, updates)
    #     return params, logs, opt_state
    
    @jax.jit
    def train_step(opt, batch):
        def loss_fn(params):
            _, z, logdets, priors = model.apply(params, batch, reverse=False)
            logpx, logpz, logdets = get_logpx(z, logdets, priors)
            return - logpx, (logpz, logdets)
        logs, grad = jax.value_and_grad(loss_fn, has_aux=True)(opt.target)
        opt = opt.apply_gradient(grad, learning_rate=lr_warmup(opt.state.step))
        return logs, opt
    
    # Helper functions for evaluation 
    @jax.jit
    def eval_step(params, batch):
        _, z, logdets, priors = model.apply(params, batch, reverse=False)
        return - get_logpx(z, logdets, priors)[0]
    
    # Helper function for sampling from random latent fixed during training for comparison
    eps = []
    for i in range(L):
        expected_h = image_size // 2**(i + 1)
        expected_c = num_channels * 2**(i + 1)
        if i == L - 1: expected_c *= 2
        eps.append(jax.random.normal(key, (num_samples, expected_h, expected_h, expected_c)))
    sample_fn = partial(sample, eps=eps, key=key, display=False,
                        sampling_temperature=sampling_temperature,
                        postprocess_fn=partial(postprocess, num_bits=num_bits))
    
    # Train
    print("Start training...")
    print("Available jax devices:", jax.devices())
    print()
    bits = 0.
    start = time.time()
    try:
        for epoch in range(num_epochs):
            # train
            for i in range(steps_per_epoch):
                batch = next(train_ds)
                loss, opt = train_step(opt, batch)
                print(f"\r\033[92m[Epoch {epoch + 1}/{num_epochs}]\033[0m"
                      f"\033[93m[Batch {i + 1}/{steps_per_epoch}]\033[0m"
                      f" loss = {loss[0]:.5f},"
                      f" (log(p(z)) = {loss[1][0]:.5f},"
                      f" logdet = {loss[1][1]:.5f})", end='')
                if np.isnan(loss[0]):
                    print("\nModel diverged - NaN loss")
                    return None, None
                
                step = epoch * steps_per_epoch + i + 1
                if step % int(num_sample_epochs * steps_per_epoch) == 0:
                    sample_fn(model, opt.target, 
                              save_path=(fp + f"step_{step:05d}.png"))

            # eval on one batch of validation samples 
            # + generate random sample
            t = time.time() - start
            if val_ds is not None:
                bits = eval_step(opt.target, next(val_ds))
                wandb.log({"training bpd":loss[0],"log(p(z))":loss[1][0],"logdet":loss[1][1],"val_bpd":bits,"epoch":epoch})
            print(f"\r\033[92m[Epoch {epoch + 1}/{num_epochs}]\033[0m"
                  f"[{int(t // 3600):02d}h {int((t % 3600) // 60):02d}mn]"
                  f" train_bits/dims = {loss[0]:.3f},"
                  f" val_bits/dims = {bits:.3f}" + " " * 50)

            # Save parameters
            if (epoch + 1) % num_save_epochs == 0 or epoch == num_epochs - 1:
                with open((weights_loc+f'model_epoch={epoch + 1:03d}.weights'), 'wb') as f:
                    f.write(flax.serialization.to_bytes(opt.target))
    
    except KeyboardInterrupt:
        print(f"\nInterrupted by user at epoch {epoch + 1}")
        
    # returns final model and parameters
    return model, opt.target


num_images = len(glob.glob(f"{config_dict['image_path']}/*.jpg"))
config_dict['steps_per_epoch'] = num_images // config_dict['batch_size']
train_split = int(config_dict['train_split'] * num_images)
val_split = (num_images-train_split)
print(f"{train_split} training images")
print(f"{val_split} validation images")
print(f"{config_dict['steps_per_epoch']} training steps per epoch")

data = tf.data.Dataset.list_files(f"{config_dict['image_path']}/*.jpg")
data = data.shuffle(buffer_size=800)

# Get the training data and skip the first 20% of the data
train_ds = get_train_dataset(data,**config_dict, skip=val_split)

# Val data
# During training we'll only evaluate on one batch of validation to save on computations
# takes the first 20% of the data that the training set skipped
val_ds = get_val_dataset(data,**config_dict, take=val_split, repeat=True)

# Sample
plot_image_grid(postprocess(next(val_ds), num_bits=config_dict['num_bits'])[:25], 
  title="Input data sample",display=False)

model, params = train_glow(train_ds, val_ds=val_ds, **config_dict)

print("Random samples evolution during training")

# filepaths
fp_in = fp + "step_*.png"
fp_out = "../sample_evolution_" + args.wb_name + ".gif"

li_imgs = [np.asarray(Image.open(f)) for f in sorted(glob.glob(fp_in))]
wandb.log({"Samples during Training": [wandb.Image(img) for img in li_imgs]})

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=200, loop=0)

rand_imgs = (np.asarray(f) for f in imgs)
wandb.log({"Samples during Training": [wandb.Image(img) for img in rand_imgs]})

sample(model, params, shape=(16,) + config_dict["sampling_shape"],  key=random_key,
       postprocess_fn=partial(postprocess, num_bits=config_dict["num_bits"]),
       save_path=(results_loc+"final_random_sample_T=1.png"));

sample(model, params, shape=(16,) + config_dict["sampling_shape"], 
       key=jax.random.PRNGKey(1), sampling_temperature=0.7,
       postprocess_fn=partial(postprocess, num_bits=config_dict["num_bits"]),
       save_path=(results_loc+"final_random_sample_T=0.7_1.png"));

sample(model, params, shape=(16,) + config_dict["sampling_shape"], 
       key=jax.random.PRNGKey(2), sampling_temperature=0.7,
       postprocess_fn=partial(postprocess, num_bits=config_dict["num_bits"]),
       save_path=(results_loc+"final_random_sample_T=0.7_2.png"))

sample(model, params, shape=(16,) + config_dict["sampling_shape"], 
       key=jax.random.PRNGKey(3), sampling_temperature=0.5,
       postprocess_fn=partial(postprocess, num_bits=config_dict["num_bits"]),
       save_path=(results_loc+"final_random_sample_T=0.5.png"))

sample_imgs = [np.asarray(Image.open(f)) for f in sorted(glob.glob((results_loc+"final_random_sample_T*.png")))]
wandb.log({"Random samples": [wandb.Image(img) for img in sample_imgs]})