def train():
    
    print('Jax version', jax.__version__)
    print('Flax version', flax.__version__)
    random_key = jax.random.PRNGKey(0)

    print("Available devices:", jax.devices()) 

    def squeeze(x):
        x = jnp.reshape(x, (x.shape[0], 
                            x.shape[1] // 2, 2, 
                            x.shape[2] // 2, 2,
                            x.shape[-1]))
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        x = jnp.reshape(x, x.shape[:3] + (4 * x.shape[-1],))
        return x
    
    print("Example")
    x = jax.random.randint(random_key, (1, 4, 4, 1), 0, 10)
    print('x = ', '\n     '.join(' '.join(str(v[0]) for v in row) for row in x[0]))
    print('\nbecomes\n')
    x = squeeze(x)
    print('y with shape', x.shape, 'where')
    print('\n'.join(f'  y[{i}, {j}] = {x[0, i, j]}' for i in range(2) for j in range(2)))
    
    def unsqueeze(x):
        x = jnp.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 
                            2, 2, x.shape[-1] // 4))
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        x = jnp.reshape(x, (x.shape[0], 
                            2 * x.shape[1],
                            2 * x.shape[3],
                            x.shape[5]))
        return x

    print("Sanity check for  reversibility")
    def sanity_check():
        x = jax.random.randint(random_key, (1, 4, 4, 16), 0, 10)
        y = unsqueeze(squeeze(x))
        z = squeeze(unsqueeze(x))
        print("  \033[92m✓\033[0m" if np.array_equal(x, y) else "  \033[91mx\033[0m", 
            "unsqueeze o squeeze = id")
        print("  \033[92m✓\033[0m" if np.array_equal(x, z) else "  \033[91mx\033[0m", 
            "squeeze o unsqueeze = id")
    sanity_check()
    
    def split(x):
        return jnp.split(x, 2, axis=-1)

    def unsplit(x, z):
        return jnp.concatenate([z, x], axis=-1)
    
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
    
    class AffineCoupling(nn.Module):
        out_dims: int
        width: int = 512
        eps: float = 1e-8
    
        @nn.compact
        def __call__(self, inputs, logdet=0, reverse=False):
            # Split
            xa, xb = jnp.split(inputs, 2, axis=-1)
            
            # NN
            net = nn.Conv(features=self.width, kernel_size=(3, 3), strides=(1, 1),
                        padding='same', name="ACL_conv_1")(xb)
            net = nn.relu(net)
            net = nn.Conv(features=self.width, kernel_size=(1, 1), strides=(1, 1),
                        padding='same', name="ACL_conv_2")(net)
            net = nn.relu(net)
            net = ConvZeros(self.out_dims, name="ACL_conv_out")(net)
            mu, logsigma = jnp.split(net, 2, axis=-1)
            # See https://github.com/openai/glow/blob/master/model.py#L376
            # sigma = jnp.exp(logsigma)
            sigma = jax.nn.sigmoid(logsigma + 2.)
            
            # Merge
            if not reverse:
                ya = sigma * xa + mu
                logdet += jnp.sum(jnp.log(sigma), axis=(1, 2, 3))
            else:
                ya = (xa - mu) / (sigma + self.eps)
                logdet -= jnp.sum(jnp.log(sigma), axis=(1, 2, 3))
                
            y = jnp.concatenate
            
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
            logdet_factor = reduce(
                operator.mul, (inputs.shape[i] for i in range(1, len(inputs.shape) - 1)), 1)
            if not reverse:
                y = sigma * (inputs + mu)
                logdet += logdet_factor * jnp.sum(logsigma)
            else:
                y = inputs / (sigma + self.eps) - mu
                logdet -= logdet_factor * jnp.sum(logsigma)
            
            # Logdet and return
            return y, logdet
        
    print("Sanity check for data-dependant init in ActNorm")

    def sanity_check():
        x = jax.random.normal(random_key, (1, 256, 256, 3))
        model = ActNorm()
        init_variables = model.init(random_key, x)
        y, _ = model.apply(init_variables, x)
        m = jnp.mean(y); v = jnp.std(y); eps = 1e-5
        print("  \033[92m✓\033[0m" if abs(m) < eps else "  \033[91mx\033[0m", "Mean:", m)
        print("  \033[92m✓\033[0m" if abs(v  - 1) < eps else "  \033[91mx\033[0m",
            "Standard deviation", v)
    sanity_check()
    
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

    class FlowStep(nn.Module):
        nn_width: int = 512
        key: jax.random.PRNGKey = jax.random.PRNGKey(0)
        
        @nn.compact
        def __call__(self, x, logdet=0, reverse=False):
            out_dims = x.shape[-1]
            if not reverse:
                x, logdet = ActNorm()(x, logdet=logdet, reverse=False)
                x, logdet = Conv1x1(out_dims, self.key)(x, logdet=logdet, reverse=False)
                x, logdet = AffineCoupling(out_dims, self.nn_width)(x, logdet=logdet, reverse=False)
            else:
                x, logdet = AffineCoupling(out_dims, self.nn_width)(x, logdet=logdet, reverse=True)
                x, logdet = Conv1x1(out_dims, self.key)(x, logdet=logdet, reverse=True)
                x, logdet = ActNorm()(x, logdet=logdet, reverse=True)
            return x, logdet
        
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
                
    # Summarize a flow step
    def summary():
        x = jax.random.normal(random_key, (32, 10, 10, 6))
        model = FlowStep(key=random_key)
        init_variables = model.init(random_key, x)
        summarize_jax_model(init_variables, max_depth=2)
    summary()
    
    class GLOW(nn.Module):
        K: int = 32                                       # Number of flow steps
        L: int = 3                                        # Number of scales
        nn_width: int = 512                               # NN width in Affine Coupling Layer
        learn_top_prior: bool = False                     # If true, learn prior N(mu, sigma) for zL
        key: jax.random.PRNGKey = jax.random.PRNGKey(0)
        
            
        def flows(self, x, logdet=0, reverse=False, name=""):
            """K subsequent flows. Called at each scale."""
            for k in range(self.K):
                it = k + 1 if not reverse else self.K - k
                x, logdet = FlowStep(self.nn_width, self.key, name=f"{name}/step_{it}")(
                    x, logdet=logdet, reverse=reverse)
            return x, logdet
            
        
        @nn.compact
        def __call__(self, x, reverse=False, z=None, eps=None, sampling_temperature=1.0):
            """Args:
                * x: Input to the model
                * reverse: Whether to apply the model or its inverse
                * z (reverse = True): If given, use these as intermediate latents (deterministic)
                * eps (reverse = True, z!=None): If given, use these as Gaussian samples which are later 
                    rescaled by the mean and variance of the appropriate prior.
                * sampling_temperature (reverse = True, z!=None): Sampling temperature
            """
            
            ## Inputs
            # Forward pass: Save priors for computing loss
            # Optionally save zs (only used for sanity check of reversibility)
            priors = []
            if not reverse:
                del z, eps, sampling_temperature
                z = []
            # In reverse mode, either use the given latent z (deterministic)
            # or sample them. For the first one, uses the top prior.
            # The intermediate latents are sampled in the `Split(reverse=True)` calls
            else:
                if z is not None:
                    assert len(z) == self.L
                else:
                    x *= sampling_temperature
                    if self.learn_top_prior:
                        # Assumes input x is a sample from N(0, 1)
                        # Note: the inputs to learn the top prior is zeros (unconditioned)
                        # or some conditioning e.g. class information.
                        # If not learnable, the model just uses the input x directly
                        # see https://github.com/openai/glow/blob/master/model.py#L109
                        prior = ConvZeros(x.shape[-1] * 2, name="prior_top")(jnp.zeros(x.shape))
                        mu, logsigma = jnp.split(prior, 2, axis=-1)
                        x = x * jnp.exp(logsigma) + mu
                    
            ## Multi-scale model
            logdet = 0
            for l in range(self.L):
                # Forward
                if not reverse:
                    x = squeeze(x)
                    x, logdet = self.flows(x, logdet=logdet,
                                        reverse=False,
                                        name=f"flow_scale_{l + 1}/")
                    if l < self.L - 1:
                        zl, x, prior = Split(
                            key=self.key, name=f"flow_scale_{l + 1}/")(x, reverse=False)
                    else:
                        zl, prior = x, None
                        if self.learn_top_prior:
                            prior = ConvZeros(zl.shape[-1] * 2, name="prior_top")(jnp.zeros(zl.shape))
                    z.append(zl)
                    priors.append(prior)
                        
                # Reverse
                else:
                    if l > 0:
                        x = Split(key=self.key, name=f"flow_scale_{self.L - l}/")(
                            x, reverse=True, 
                            z=z[-l - 1] if z is not None else None,
                            eps=eps[-l - 1] if eps is not None else None,
                            temperature=sampling_temperature)
                    x, logdet = self.flows(x, logdet=logdet, reverse=True,
                                        name=f"flow_scale_{self.L - l}/")
                    x = unsqueeze(x)
                    
            ## Return
            return x, z, logdet, priors

    print("Sanity check for reversibility (no sampling in reverse pass)")

    def sanity_check():
        # Input
        x_1 = jax.random.normal(random_key, (32, 32, 32, 6))
        K, L = 16, 3
        model = GLOW(K=K, L=L, nn_width=128, key=random_key, learn_top_prior=True)
        init_variables = model.init(random_key, x_1)

        # Forward call
        _, z, logdet, priors = model.apply(init_variables, x_1)

        # Check output shape
        expected_h = x_1.shape[1] // 2**L
        expected_c = x_1.shape[-1] * 4**L // 2**(L - 1)
        print("  \033[92m✓\033[0m" if z[-1].shape[1] == expected_h and z[-1].shape[-1] == expected_c 
            else "  \033[91mx\033[0m",
            "Forward pass output shape is", z[-1].shape)

        # Check sizes of the intermediate latent
        correct_latent_shapes = True
        correct_prior_shapes = True
        for i, (zi, priori) in enumerate(zip(z, priors)):
            expected_h = x_1.shape[1] // 2**(i + 1)
            expected_c = x_1.shape[-1] * 2**(i + 1)
            if i == L - 1:
                expected_c *= 2
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
    sanity_check()
    
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

    @jax.jit
    def postprocess(x, num_bits):
        """Map [-0.5, 0.5] quantized images to uint space"""
        num_bins = 2 ** num_bits
        x = jnp.floor((x + 0.5) * num_bins)
        x *= 256. / num_bins
        return jnp.clip(x, 0, 255).astype(jnp.uint8)
    
    def sample(model, 
            params, 
            eps=None, 
            shape=None, 
            sampling_temperature=1.0, 
            key=jax.random.PRNGKey(0),
            postprocess_fn=None, 
            save_path=None,
            display=True):
        """Sampling only requires a call to the reverse pass of the model"""
        if eps is None:
            zL = jax.random.normal(key, shape) 
        else: 
            zL = eps[-1]
        y, *_ = model.apply(params, zL, eps=eps, sampling_temperature=sampling_temperature, reverse=True)
        if postprocess_fn is not None:
            y = postprocess_fn(y)
        plot_image_grid(y, save_path=save_path, display=display,
                        title=None if save_path is None else save_path.rsplit('.', 1)[0].rsplit('/', 1)[1])
        return y


    from mpl_toolkits.axes_grid1 import ImageGrid
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
    
    def train_glow(train_ds,
               val_ds=None,
               num_samples=9,
               image_size=256,
               num_channels=3,
               num_bits=5,
               init_lr=1e-3,
               num_epochs=1,
               num_sample_epochs=1,
               num_warmup_epochs=10,
               num_save_epochs=1,
               steps_per_epoch=1,
               K=32,
               L=3,
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
            num_save_epochs: save mode at this interval
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
                    nn_width=nn_width, 
                    learn_top_prior=learn_top_prior,
                    key=key)
        
        # Init optimizer and learning rate schedule
        params = model.init(random_key, next(train_ds))
        opt = flax.optim.Adam(learning_rate=init_lr).create(params)
        
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
        if not os.path.exists("samples"): os.makedirs("samples")
        if not os.path.exists("weights"): os.makedirs("weights")
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
                                save_path=f"samples/step_{step:05d}.png")

                # eval on one batch of validation samples 
                # + generate random sample
                t = time.time() - start
                if val_ds is not None:
                    bits = eval_step(opt.target, next(val_ds))
                print(f"\r\033[92m[Epoch {epoch + 1}/{num_epochs}]\033[0m"
                    f"[{int(t // 3600):02d}h {int((t % 3600) // 60):02d}mn]"
                    f" train_bits/dims = {loss[0]:.3f},"
                    f" val_bits/dims = {bits:.3f}" + " " * 50)
                
                # Save parameters
                if (epoch + 1) % num_save_epochs == 0 or epoch == num_epochs - 1:
                    with open(f'weights/model_epoch={epoch + 1:03d}.weights', 'wb') as f:
                        f.write(flax.serialization.to_bytes(opt.target))
        except KeyboardInterrupt:
            print(f"\nInterrupted by user at epoch {epoch + 1}")
            
        # returns final model and parameters
        return model, opt.target
    
    config_dict = {
        'image_path': "input\celeba-dataset\img_align_celeba\img_align_celeba",
        'train_split': 0.6,
        'image_size': 64,
        'num_channels': 3,
        'num_bits': 5,
        'batch_size': 64,
        'K': 16,
        'L': 3,
        'nn_width': 512, 
        'learn_top_prior': True,
        'sampling_temperature': 0.7,
        'init_lr': 1e-3,
        'num_epochs': 13,
        'num_warmup_epochs': 1,
        'num_sample_epochs': 0.2, # Fractional epochs for sampling because one epoch is quite long 
        'num_save_epochs': 5,
    }

    output_hw = config_dict["image_size"] // 2 ** config_dict["L"]
    output_c = config_dict["num_channels"] * 4**config_dict["L"] // 2**(config_dict["L"] - 1)
    config_dict["sampling_shape"] = (output_hw, output_hw, output_c)

    import glob
    import tensorflow as tf
    import tensorflow_datasets as tfds
    tf.config.experimental.set_visible_devices([], 'GPU')

    def get_train_dataset(image_path, image_size, num_bits, batch_size, skip=None, **kwargs):
        del kwargs
        train_ds = tf.data.Dataset.list_files(f"{image_path}\*.jpg")
        if skip is not None:
            train_ds = train_ds.skip(skip)
        train_ds = train_ds.shuffle(buffer_size=20000)
        train_ds = train_ds.map(partial(map_fn, size=image_size, num_bits=num_bits, training=True))
        train_ds = train_ds.batch(batch_size)
        train_ds = train_ds.repeat()
        return iter(tfds.as_numpy(train_ds))


    def get_val_dataset(image_path, image_size, num_bits, batch_size, 
                        take=None, repeat=False, **kwargs):
        del kwargs
        val_ds = tf.data.Dataset.list_files(f"{image_path}\*.jpg")
        if take is not None:
            val_ds = val_ds.take(take)
        val_ds = val_ds.map(partial(map_fn, size=image_size, num_bits=num_bits, training=False))
        val_ds = val_ds.batch(batch_size)
        if repeat:
            val_ds = val_ds.repeat()
        return iter(tfds.as_numpy(val_ds))
    

    num_images = len(glob.glob(f"{config_dict['image_path']}\*.jpg"))
    config_dict['steps_per_epoch'] = num_images // config_dict['batch_size']
    train_split = int(config_dict['train_split'] * num_images)
    print(f"{num_images} training images")
    print(f"{config_dict['steps_per_epoch']} training steps per epoch")

    #Train data
    train_ds = get_train_dataset(**config_dict, skip=train_split)

    # Val data
    # During training we'll only evaluate on one batch of validation 
    # to save on computations
    val_ds = get_val_dataset(**config_dict, take=config_dict['batch_size'], repeat=True)

    # Sample
    plot_image_grid(postprocess(next(val_ds), num_bits=config_dict['num_bits'])[:25], 
                    title="Input data sample")
    print("Image Plotted.")
    
    model, params = train_glow(train_ds, val_ds=val_ds, **config_dict)
    
if __name__ == '__main__' :

    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]= '0.90'
    import jax
    import flax
    import jax.numpy as jnp
    import flax.linen as nn

    import tensorflow as tf
    import time
    import operator
    from functools import partial
    from functools import reduce

    import numpy as np
    from matplotlib import pyplot as plt
    train()
