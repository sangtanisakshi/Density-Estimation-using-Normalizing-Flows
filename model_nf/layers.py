import operator
import jax
import flax
import jax.numpy as jnp
import flax.linen as nn
import random
import utils
import wandb
from functools import reduce
from einops import rearrange

### From one scale to another: split / unsplit, with learnable prior
class ConvZeros(nn.Module):
    features: int
        
    @nn.compact
    def __call__(self, x, logscale_factor=3.0):
        """A simple convolutional layers initializer to all zeros"""
        x = nn.Conv(self.features, kernel_size=(3, 3),
                    strides=(1, 1), padding='same',
                    kernel_init=jax.nn.initializers.zeros,
                    bias_init=jax.nn.initializers.zeros)(x)
        return x


class Split(nn.Module):
    key: jax.random.PRNGKey = jax.random.PRNGKey(0)
        
    @nn.compact
    def __call__(self, x, reverse=False, z=None, eps=None, temperature=1.0):
        """Args (reverse = True):
            * z: If given, it is used instead of sampling (= deterministic mode).
                This is only used to test the reversibility of the model.
            * eps: If z is None and eps is given, then eps is assumed to be a 
                sample from N(0, 1) and rescaled by the mean and variance of 
                the prior. This is used during training to observe how sampling
                from fixed latents evolve. 
               
        If both are None, the model samples z from scratch
        """
        if not reverse:
            del z, eps, temperature
            z, x = jnp.split(x, 2, axis=-1)
            
        # Learn the prior parameters for z
        prior = ConvZeros(x.shape[-1] * 2, name="conv_prior")(x)
            
        # Reverse mode: Only return the output
        if reverse:
            # sample from N(0, 1) prior (inference)
            if z is None:
                if eps is None:
                    eps = jax.random.normal(self.key, x.shape) 
                eps *= temperature
                mu, logsigma = jnp.split(prior, 2, axis=-1)
                z = eps * jnp.exp(logsigma) + mu
            return jnp.concatenate([z, x], axis=-1)
        # Forward mode: Also return the prior as it is used to compute the loss
        else:
            return z, x, prior
         
### Affine Coupling 
class AffineCoupling(nn.Module):
    out_dims : int
    width: int = 512
    eps:float = 1e-8
    ##Get neighbours of the selected patch
        
    @nn.compact
    def __call__(self, inputs, logdet=0, reverse=False, dilation=False, only_neighbours=True):
        
        patch_indices = list(range(0,16,1))
        random_patch_indices = random.sample(population=patch_indices,k=1)
        rest_indices = [idx for idx in patch_indices if idx not in random_patch_indices]
        ps = utils.get_patch_size(inputs, dilation)
        
        #does not work with multiple patches for obvious reasons.
        if only_neighbours:
            neighbours = utils.get_neighbours(random_patch_indices) 
            
        # We now divide each image into patches
        if not dilation:
            all_patches = rearrange(inputs, 'b (nh hp) (nw wp) c -> b (nh nw) hp wp c ', hp=ps, wp=ps)
        else:
            all_patches = rearrange(inputs, 'b (nh hp) (nw wp) c -> b (hp wp) nh nw c ', hp=ps, wp=ps)#dilated convolution
            
        network = jnp.zeros((inputs.shape[0],len(random_patch_indices),all_patches.shape[2],all_patches.shape[2],6))
        chosen_patches = all_patches[:,(random_patch_indices),:,:,:] #shape = (batch_size,patches,8,8,3)
        
        ACL_conv1 = nn.Conv(features=self.width, kernel_size=(3, 3), strides=(1, 1),
                        padding='same', name=("ACL_conv_1")) # o/p shape = (batch_size,8,8,self.width)
        ACL_conv2 = nn.Conv(features=self.width, kernel_size=(1, 1), strides=(1, 1),
                        padding='same', name=("ACL_conv_2"))
        ACL_conv0 = ConvZeros((self.out_dims*2), name=("ACL_conv0"))# o/p shape = (batch_size,8,8,6) so that the split of mu and logsigma can make the element wise multiplication possible
        
        for patch in range(chosen_patches.shape[1]):
            net = ACL_conv1(chosen_patches[:,patch,:,:,:])
            net = nn.relu(net)
            net = ACL_conv2(net)
            net = nn.relu(net)
            net = ACL_conv0(net)
            network.at[:,patch,:,:,:].set(net)
        
        mu, logsigma = jnp.split(network, 2, axis=-1) # mu and logsigma will be also the same dimension as the to-be-transformed
        sigma = jax.nn.sigmoid(logsigma + 2.)
        mu = jnp.average(mu,axis=1)
        sigma = jnp.average(sigma,axis=1)
        
        if not only_neighbours:
            if not reverse:
                for index in rest_indices:
                    all_patches = all_patches.at[:,index,:,:,:].multiply(sigma)
                    all_patches = all_patches.at[:,index,:,:,:].add(mu)
                sigma = jnp.expand_dims(sigma, axis=1)
                sigma = sigma.repeat(len(rest_indices),axis=1)
                logdet += jnp.sum(jnp.log(sigma), axis=(1, 2, 3, 4))
            else:
                for index in rest_indices:
                    all_patches = all_patches.at[:,index,:,:,:].add(-mu)
                    all_patches = all_patches.at[:,index,:,:,:].divide(sigma+self.eps)
                sigma = jnp.expand_dims(sigma, axis=1)
                sigma = sigma.repeat(len(rest_indices),axis=1)
                logdet -= jnp.sum(jnp.log(sigma), axis=(1, 2, 3, 4))
            y = all_patches
                 
        else:
            if not reverse:
                for index in neighbours:
                    all_patches = all_patches.at[:,index,:,:,:].multiply(sigma)
                    all_patches = all_patches.at[:,index,:,:,:].add(mu)
                sigma = sigma.repeat(len(neighbours),axis=1)
                logdet += jnp.sum(jnp.log(sigma), axis=(1, 2, 3))
            else:
                for index in neighbours:
                    all_patches = all_patches.at[:,index,:,:,:].add(-mu)
                    all_patches = all_patches.at[:,index,:,:,:].divide(sigma+self.eps)
                sigma = sigma.repeat(len(neighbours),axis=1)
                logdet = jnp.sum(jnp.log(sigma), axis=(1, 2, 3))
            y = all_patches   

        #Turn patches back to the image - turn back y into (256, 32, 32, 3)
        if not dilation:
            y = rearrange(y, 'b (nh nw) hp wp c -> b (nh hp) (nw wp) c ', nh=4, nw=4) #turn back y into (256, 32, 32, 3)
        else:
            y = rearrange(y, 'b (hp wp) nh nw c -> b (nh hp) (nw wp) c ', hp=4, wp=4) #dilated convolution
            
        return y, logdet

### Activation Normalization
class ActNorm(nn.Module):
    scale: float = 1.
    eps: float = 1e-8

    @nn.compact
    def __call__(self, inputs, logdet=0, reverse=False):
        # Data dependent initialization. Will use the values of the batch
        # given during model.init
        axes = tuple(i for i in range(len(inputs.shape) - 1))
        def dd_mean_initializer(key, shape):
            """Data-dependant init for mu"""
            nonlocal inputs
            x_mean = jnp.mean(inputs, axis=axes, keepdims=True)
            return - x_mean
        
        def dd_stddev_initializer(key, shape):
            """Data-dependant init for sigma"""
            nonlocal inputs
            x_var = jnp.mean(inputs**2, axis=axes, keepdims=True)
            var = self.scale / (jnp.sqrt(x_var) + self.eps)
            return var
        
        # Forward
        shape = (1,) * len(axes) + (inputs.shape[-1],)
        mu = self.param('actnorm_mean', dd_mean_initializer, shape)
        sigma = self.param('actnorm_sigma', dd_stddev_initializer, shape)
        logsigma = jnp.log(jnp.abs(sigma))
     #   logdet_factor = reduce(
     #       operator.mul, (inputs.shape[i] for i in range(1, len(inputs.shape) - 1)), 1)
        logdet_factor = 1
        if not reverse:
            y = sigma * (inputs + mu)
            logdet += logdet_factor * jnp.sum(logsigma)
        else:
            y = inputs / (sigma + self.eps) - mu
            logdet -= logdet_factor * jnp.sum(logsigma)
        
        # Logdet and return
        return y, logdet
    
    
### Invertible 1x1 Convolution
class Conv1x1(nn.Module):
    channels: int
    key: jax.random.PRNGKey = jax.random.PRNGKey(0)

    def setup(self):
        """Initialize P, L, U, s"""
        # W = PL(U + s)
        # Based on https://github.com/openai/glow/blob/master/model.py#L485
        c = self.channels
        # Sample random rotation matrix
        q, _ = jnp.linalg.qr(jax.random.normal(self.key, (c, c)), mode='complete')
        p, l, u = jax.scipy.linalg.lu(q)
        # Fixed Permutation (non-trainable)
        self.P = p
        self.P_inv = jax.scipy.linalg.inv(p)
        # Init value from LU decomposition
        L_init = l
        U_init = jnp.triu(u, k=1)
        s = jnp.diag(u)
        self.sign_s = jnp.sign(s)
        S_log_init = jnp.log(jnp.abs(s))
        self.l_mask = jnp.tril(jnp.ones((c, c)), k=-1)
        self.u_mask = jnp.transpose(self.l_mask)
        # Define trainable variables
        self.L = self.param("L", lambda k, sh: L_init, (c, c))
        self.U = self.param("U", lambda k, sh: U_init, (c, c))
        self.log_s = self.param("log_s", lambda k, sh: S_log_init, (c,))
        
        
    def __call__(self, inputs, logdet=0, reverse=False):
        c = self.channels
        assert c == inputs.shape[-1]
        # enforce constraints that L and U are triangular
        # in the LU decomposition
        L = self.L * self.l_mask + jnp.eye(c)
        U = self.U * self.u_mask + jnp.diag(self.sign_s * jnp.exp(self.log_s))
        logdet_factor = inputs.shape[1] * inputs.shape[2]
        
        # forward
        if not reverse:
            # lax.conv uses weird ordering: NCHW and OIHW
            W = jnp.matmul(self.P, jnp.matmul(L, U))
            y = jax.lax.conv(jnp.transpose(inputs, (0, 3, 1, 2)), 
                             W[..., None, None], (1, 1), 'same')
            y = jnp.transpose(y, (0, 2, 3, 1))
            logdet += jnp.sum(self.log_s) * logdet_factor
        # inverse
        else:
            W_inv = jnp.matmul(jax.scipy.linalg.inv(U), jnp.matmul(
                jax.scipy.linalg.inv(L), self.P_inv))
            y = jax.lax.conv(jnp.transpose(inputs, (0, 3, 1, 2)),
                             W_inv[..., None, None], (1, 1), 'same')
            y = jnp.transpose(y, (0, 2, 3, 1))
            logdet -= jnp.sum(self.log_s) * logdet_factor
            
        return y, logdet