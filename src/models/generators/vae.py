"""
    src/models/gens/vae.py
    ----------------------

    This module implements a variational autoencoder (Vae) in 
    TensorFlow/Keras. The Vae is a generative model that learns 
    a latent representation of data, by a probabilistic 
    encoding/decoding. It consists of:
        
        - **Encoder**:  Maps input data and conditioning variables 
                        to a latent gaussian distribution 
                        parameterized by mean (`µ`) and log-variance
                        (`log(σ²)`).
        - **Reparameterization-Layer**: Samples latent codes `z` using
                                        the "_reparameterization trick_"
                                        to enable gradient-based
                                        optimization.
        - **Decoder**:  Maps sampled latent codes (and conditions) back
                        into reconstructed data, resemble input data to the
                        encoder
        - **Loss-Functions**:   Combines _Gaussian_ reconstruction loss and 
                                KL-Divergence regularization, with optional
                                β-weighting and annealing.
    
    Key Features within this Version of VAE:
        - Customizable encoder/encoder depth and width
        - KL-annealing and β-VAE support
    
    Used in this project as a core module in path modeling, particular used 
    in learning data and reconstruct also generate new trajectories based on 
    the conditions (environmental conditions) as well as the link state passed
    by the link model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers

from src.models.utils.helpers import (
    extract_inputs, set_initialization,
    SplitSortLayer
)
from src.config.const import WEIGHTS_FN, CONFIG_FN


# --------------------------- Reparameterization --------------------------- #

class Reparameterize(tfkl.Layer):
    """
        Reparameterization trick implementation for VAEs.

        Given _Gaussian_ parameters (`µ`, `log(σ²)`), this layer samples 
        latent vectors, in form of `z = µ + σ ⊙ ε`, where `ε~N(O,I)`. 
        This makes sampling differentiable and allows gradients to 
        propagate through stochastic nodes.

        Args:
        -----
            inputs: Tuple[mu, logvar]
                - mu:   Tensor of shape (`batch_size`, `latent_dim`), 
                        mean of latent Gaussian.
                - logvar:   Tensor of same shape as mean, log-variance
                            of the latent Gaussian.
        
        Returns:
        --------
            Tensor of shape (`batch_size`, `latent_dim`), sampled latent
            vectors.
        --------
    """
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        mu, logvar = inputs
        eps = tf.random.normal(shape=tf.shape(mu), dtype=mu.dtype)

        return mu + eps * tf.exp(0.50 * logvar)
    
    def get_config(self): return super().get_config()


# --------------------------------- Losses --------------------------------- #

def reconstruction_loss(x: tf.Tensor, 
                        mu: tf.Tensor, 
                        logvar: tf.Tensor) -> tf.Tensor:
    """
        Gaussian reconstruction loss with diagonal covariance included.

        Computes the negative log-likelihood of observed data under the
        reconstructed Gaussian distribution
        
        L_rec = 0.5 * Σ_j [ precision_j * (x_j - μ_j)² + log σ²_j ]
        
        Averaged across the batch.

        Args:
        -----
            x:  Tensor of shape (`batch_size`, `n_features`), original data
            mu: Tensor of shape, reconstruction mean.
            logvar: Tensor of same shape, reconstructed log-variance.
        
        Returns:
        --------
            Scalar tensor, mean reconstructed loss over batch.
        --------
    """
    logvar = tf.clip_by_value(logvar, -10.0, 10.0)
    precision = tf.exp(-logvar)

    # 0.50 · Σ(precision · (x-µ)² + log(σ)) --average over batch
    per_sample = 0.50 * tf.reduce_sum(precision * tf.square(x - mu) + logvar,
                                      axis=-1)
    return tf.reduce_mean(per_sample)


def kl_divergence(mu: tf.Tensor, 
                  logvar: tf.Tensor, 
                  weights: tf.Tensor) -> tf.Tensor:
    """
        KL Divergence between approximate posterior `q(z|x)` and prior
        `p(z)`, it computes the alignment between the latent generated
        statistical distribution and the data distribution.

        Computes:
        ---------
            KL(q||p) = -0.5 * Σ_j[1 + log(σ²_j - µ_j² - σ²_j)]
        
        Optionally scaled by an annealing or β-weighting factor.

        Args:
        -----
            mu: Tensor of shape (`batch_size`, `latent_dim`), posterior
                mean
            logvar: Tensor of same shape as the mean `µ`, posterior 
                    log-variance
            weights:    Scalar or batch-shaped tensor, weight applied 
                        to KL term.
        
        Returns:
        --------
            Scalar tensor, mean KL divergence across the batch.
        --------
    """
    logvar = tf.clip_by_value(logvar, -10.0, 10.0)
    kl = -0.50 * tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), 
                               axis=-1)
    return tf.reduce_mean(weights * kl)





# ---------------------------------- VAE ---------------------------------- #

class Vae(tfk.Model):
    """
        Variational Autoencoder (VAE) with optional β-VAE and KL annealing.
        Constructs the encoder -> latente space -> add Noise -> decoder
        operates with noise added to the standard autoencoder.

        This class implements:
        - Encoder network: maps (x, cond) → (z_mu, z_logvar)
        - Decoder network: maps (z, cond) → (x_mu, x_logvar)
        - Reparameterization trick for latent sampling
        - Training loop with reconstruction + KL losses
        - Custom metrics for logging training progress
        - Save/load functionality with JSON config persistence

        Args:
            n_latent: Dimensionality of latent space
            n_data: Number of input data features
            n_conditions: Number of conditioning variables
            encoder_layers: Sequence of hidden layer sizes for encoder
            decoder_layers: Sequence of hidden layer sizes for decoder
            min_variance: Lower bound for output variance (numerical stability)
            dropout_rate: Dropout probability for hidden layers
            beta: Weight on KL loss (β-VAE)
            beta_annealing_step: Number of steps to linearly anneal β to 1.0
            kl_warmup_steps: Number of steps to ramp KL weight from 0 → 1
            init_kernel_std: Stddev for kernel initialization
            init_bias_std: Stddev for bias initialization
            n_sort: Optional SplitSortLayer parameter (post-decoder)
    """
    def __init__(self,
        n_latent: int, n_data: int, n_conditions: int,
        encoder_layers          : Sequence[int] = (256, 128),
        decoder_layers          : Sequence[int] = (128, 256),
        min_variance            : float = 1e-4,
        dropout_rate            : float = 0.12,
        beta                    : float = 0.50,
        beta_annealing_step     : int   = 10_000,
        kl_warmup_steps         : int   = 1_000,
        init_kernel_std         : float = 10.0,
        init_bias_std           : float = 10.0,
        n_sort                  : int   = 0,
        **kwargs
    ):
        """ Initialize the Vae Model """ 
        super().__init__(**kwargs)
        # super(Vae, self).__init__(**kwargs)

        # Hyperparameters
        self.n_latent = int(n_latent) 
        self.n_data = int(n_data) 
        self.n_conditions = int(n_conditions)

        self.encoder_layers = tuple(int(u) for u in encoder_layers)
        self.decoder_layers = tuple(int(u) for u in decoder_layers)

        self.min_variance = float(min_variance)
        self.dropout_rate = float(dropout_rate)

        self.init_kernel_std = float(init_kernel_std)
        self.init_bias_std = float(init_bias_std)
        self.n_sort = int(n_sort)

        # Schedulers
        self.beta = tf.Variable(float(beta), trainable=False, dtype=tf.float32)
        self.beta_annealing_step = int(beta_annealing_step)
        self.kl_warmup_steps = tf.constant(int(kl_warmup_steps), dtype=tf.float32)
        self.current_step = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        # Layers / Submodels
        self.sampler = Reparameterize()
        self.encoder = self._build_encoder(layers=self.encoder_layers,
                                           kernel_std=self.init_kernel_std,
                                           bias_std=self.init_bias_std,
                                           dropout_rate=self.dropout_rate)

        self.decoder = self._build_decoder(layers=self.decoder_layers,
                                           kernel_std=self.init_kernel_std,
                                           bias_std=self.init_bias_std,
                                           dropout_rate=self.dropout_rate,
                                           n_sort=n_sort)
        self._initialize_metrics()
    

    @property
    def metrics(self) -> List[tfk.metrics.Metric]:
        return [
            self.total_loss_tracker, self.recon_loss_tracker, 
            self.kl_divergence_tracker, self.beta_tracker,
            self.kl_weight_tracker, self.mse_tracker,
            self.mae_tracker
        ]


    # -------------------------------- Forward -------------------------------- #

    def call(self,
        inputs, training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
            Executes a forward pass throught the architecture of the
            neural network for the VAE,
                >>> X -> Encoder -> Z -> Decoder -> X'
            Steps:
            ------
                1.  Encoder inputs `(x, cond)` into latent distribution
                    parameteres `(z_mu, z_logvar)`.
                2.  Sample latent code `Z` using the reparametrization
                    trick,
                3.  Decoder `(z, cond)` back into reconstructed 
                    distribution `(x_mu, x_logvar)`

            Args:
            -----
                inputs: Tuple of tensors, `(x_mu, x_logvar)`.
                    - `x`:  Tensor of shape `(batch_size, n_data)`,
                            input features,
                    - `cond`:   Tensor of shape `(batch_size, n_data)`,
                                conditioning variables related to the
                                environment.
                training:   whether the call is in training mode or not
            
            Returns:
            --------
            Tuple `(x_mu, x_logvar, z_mu, z_logvar)`
                - x_mu: Reconstructed mean, shape `(batch_size, n_data)`.
                - x_logvar: Reconstructed log-variance, shape `(batch_size, n_data)`.
                - z_mu: Latent mean, shape `(batch_size, n_latent)`.
                - z_logvar: Latent log-variance, shape `(batch_size, n_latent)`.
            ---------
        """
        x, cond = extract_inputs(inputs)
        z_mu, z_logvar = self.encoder([x, cond], training=training)
        z = self.sampler([z_mu, z_logvar])
        x_mu, x_logvar = self.decoder([z, cond], training=training)

        return x_mu, x_logvar, z_mu, z_logvar


    # --------------------------- Training / Testing --------------------------- #

    def _update_schedulers(self):
        """
            Update internal scheduling variables after each step.

            - Increments the training step counter.
            - Updates β (KL weight) if `beta_annealing_step > 0`,
            gradually annealing toward 1.0.
        """
        self.current_step.assign_add(1.0)
        if self.beta_annealing_step > 0:
            self.beta.assign(tf.minimum(
                1.0, self.beta + 1.0 / self.beta_annealing_step
            ))
    

    def _kl_weight(self) -> tf.Tensor:
        """
            Compute current KL divergence weighting factor.

            - Linearly increases from 0 → 1 over `kl_warmup_steps`.
            - Used to avoid posterior collapse early in training.

            Returns:
                Scalar tensor in [0.0, 1.0], KL weight for current step.
        """
        return tf.minimum(1.0, self.current_step / self.kl_warmup_steps)


    def train_step(self, inputs):
        """
            Custom training logic for one batch.

            Procedure:
            -----------
            1. Update KL/beta schedulers.
            2. Forward pass through encoder → reparameterization → decoder.
            3. Compute losses: \
                Gaussian reconstruction + weighted KL divergence.
            4. Backpropagate and apply gradients.
            5. Update custom metrics.

            Args:
                inputs: Tuple `(x, cond)`
                    - x: Tensor of shape `(batch_size, n_data)`.
                    - cond: Tensor of shape `(batch_size, n_conditions)`.

            Returns:
                Dict[str, tf.Tensor]: Current metric values.
        """
        x, cond = extract_inputs(inputs)
        self._update_schedulers()
        kl_w = self._kl_weight()

        with tf.GradientTape() as tape:
            # call method invoked
            x_mu, x_logvar, z_mu, z_logvar = self([x, cond], training=True)
            recon = reconstruction_loss(x, x_mu, x_logvar)
            kld = kl_divergence(z_mu, z_logvar, kl_w)

            total = recon + self.beta * kld

        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self._update_metrics(x, x_mu, recon, kld, total, kl_w)

        return { metric.name: metric.result() for metric in self.metrics }


    
    def test_step(self, inputs):
        """
            Custom evaluation logic for one batch.

            Similar to `train_step` but without gradient updates:
            - Computes forward pass.
            - Calculates reconstruction + KL losses.
            - Updates metrics for reporting.

            Args:
                inputs: Tuple `(x, cond)`

            Returns:
                Dict[str, tf.Tensor]: Current metric values.
        """
        x, cond = extract_inputs(inputs)
        kl_w = self._kl_weight()
        x_mu, x_logvar, z_mu, z_logvar = self([x, cond], training=False)

        recon = reconstruction_loss(x, x_mu, x_logvar)
        kld   = kl_divergence(z_mu, z_logvar, kl_w)
        total = recon + self.beta * kld

        self._update_metrics(x, x_mu, recon, kld, total, kl_w)
        return {metric.name: metric.result() for metric in self.metrics}


    # ---------------------------------- I/O ---------------------------------- #

    def save(self, directory: Union[str, Path]):
        """
            Save model weights and configuration to directory.

            - Weights saved to WEIGHTS_FN
            - Config saved as JSON to CONFIG_FN

            Args:
            -----
                directory: Path where files are written.
            -----
        """
        # Save the weights of the model
        self.save_weights(str(directory / WEIGHTS_FN))
        config = {
            "n_latent": self.n_latent, "n_data": self.n_data, "n_conditions": self.n_conditions,
            "encoder_layers": self.encoder_layers, "decoder_layers": self.decoder_layers,
            "min_variance": self.min_variance, "dropout_rate": self.dropout_rate,
            "beta": float(self.beta.numpy()),
            "beta_annealing_step": int(self.beta_annealing_step),
            "kl_warmup_steps": int(self.kl_warmup_steps),
            "init_kernel_std": self.init_kernel_std, "init_bias_std": self.init_bias_std,
            "n_sort": self.n_sort,
        }
        with open(directory / CONFIG_FN, "w") as fp:
            json.dump(config, fp)


    @classmethod
    def load(cls, directory: Union[str, Path]) -> "Vae":
        """
            Load model weights and configuration from directory.

            Args:
            -----
                directory: Path containing saved weights/config

            Returns:
            --------
                Restored Vae model.
            --------
        """
        with open(directory / CONFIG_FN, "r") as fp:
            config = json.load(fp)

        model = cls(**config)

        # Build variable using a dummy pass
        x0 = tf.zeros((1, config["n_data"]), dtype=tf.float32)
        c0 = tf.zeros((1, config["n_conditions"]), dtype=tf.float32)
        _ = model([x0, c0], training=False)

        model.load_weights(str(directory / WEIGHTS_FN))
        return model

    
    # -------------------------------- Internals -------------------------------- #

    def _initialize_metrics(self):
        """
            Initialize all custom metrics for training/testing.

            Includes:
            - Total loss, reconstruction loss, KL divergence
            - KL/beta weights
            - Reconstruction quality metrics (MSE, MAE)
        """
        self.total_loss_tracker = tfk.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tfk.metrics.Mean(name="recon_loss")
        self.kl_divergence_tracker = tfk.metrics.Mean(name="kl_div")
        self.beta_tracker = tfk.metrics.Mean(name="beta")
        self.kl_weight_tracker = tfk.metrics.Mean(name="kl_weight")
        self.mse_tracker = tfk.metrics.MeanSquaredError(name="mse")
        self.mae_tracker = tfk.metrics.MeanAbsoluteError(name="mae")


    def _update_metrics(self,
        x: tf.Tensor, mu: tf.Tensor, recon_loss_val: tf.Tensor,
        kl_div_val: tf.Tensor, total_loss_val: tf.Tensor, weights: tf.Tensor
    ):
        """
            Update all tracked metrics for the current batch.

            Args:
            -----
                x: Ground-truth inputs, shape `(batch_size, n_data)`.
                mu: Reconstructed mean, shape `(batch_size, n_data)`.
                recon_loss_val: Scalar reconstruction loss value.
                kl_div_val: Scalar KL divergence loss value.
                total_loss_val: Scalar total loss value.
                weights: KL weight used for this step.
        """
        self.total_loss_tracker.update_state(total_loss_val)
        self.recon_loss_tracker.update_state(recon_loss_val)
        self.kl_divergence_tracker.update_state(kl_div_val)
        self.beta_tracker.update_state(self.beta)
        self.kl_weight_tracker.update_state(weights)
        self.mse_tracker.update_state(x, mu)
        self.mae_tracker.update_state(x, mu)


    def _build_encoder(self, 
        layers: Sequence[int],
        kernel_std: float, bias_std: float,
        dropout_rate: float
    ) -> tfk.Model:
        """
            Construct the encoder submodel.

            Architecture:
            --------------
            - Input: `(x, cond)` concatenated.
            - Hidden layers: dense + batchnorm + optional dropout.
            - Outputs: `(z_mu, z_logvar)` parameterizing latent Gaussian.

            Returns:
            Keras Model mapping `(x, cond) → (z_mu, z_logvar)`.
        """
        # Constructing input layer
        x_in = tfkl.Input(shape=(self.n_data,), name='enc-x')
        cond_in = tfkl.Input(shape=(self.n_conditions,), name='enc-cond')
        h = tfkl.Concatenate(name='enc-x-cond')([x_in, cond_in])

        # Constructing hidden layers
        names: List[str] = []
        for i, units in enumerate(layers):
            name=f"enc-hidden-{i}"
            names.append(name)
            h = tfkl.Dense(units=units, activation='swish', name=name)(h)
            h = tfkl.BatchNormalization(name=f"{name}-batch")(h)
            if dropout_rate > 0.0:
                h = tfkl.Dropout(dropout_rate, name=f"{name}-drop")(h)

        # Constructing output layer
        z_mu = tfkl.Dense(units=self.n_latent, activation='linear', name='enc-mu')(h)
        z_logvar = tfkl.Dense(units=self.n_latent, activation='linear', 
                              name='enc-logvar')(h)
        encoder = tfk.Model([x_in, cond_in], [z_mu, z_logvar], name='encoder')
        _ = encoder([tf.zeros((1, self.n_data)), tf.zeros((1, self.n_conditions))])

        # Assign custom initialization for weights
        set_initialization(model=encoder, names=names, kernel_init=kernel_std,
                           bias_init=bias_std)
        return encoder


    def _build_decoder(self,
        layers: Sequence[int],
        kernel_std: float, bias_std: float,
        dropout_rate: float, n_sort: int
    ) -> tfk.Model:
        """
            Construct the decoder submodel.

            Architecture:
            --------------
            - Input: `(z, cond)` concatenated.
            - Hidden layers: dense + batchnorm + optional dropout.
            - Outputs:
                - x_mu: Reconstruction mean (with optional SplitSortLayer).
                - x_logvar: Reconstruction log-variance, stabilized with
                `softplus + min_variance`.

            Returns:
                Keras Model mapping `(z, cond) → (x_mu, x_logvar)`.
        """
        # Constructing input layer
        z_in = tfkl.Input(shape=(self.n_latent,), name='dec-z')
        cond_in = tfkl.Input(shape=(self.n_conditions,), name='dec-cond')
        h = tfkl.Concatenate(name='dec-concat')([z_in, cond_in])

        # Constructing hidden layers
        names: List[str] = []
        for i, units in enumerate(layers):
            name = f"dec-hidden-{i}"
            names.append(name)
            h = tfkl.Dense(units=units, activation='tanh', name=name)(h)
            h = tfkl.BatchNormalization(name=f"{name}-batch")(h)
            if dropout_rate > 0.0:
                h = tfkl.Dropout(dropout_rate, name=f"{name}-drop")(h)

        # Mean Head
        x_mu = tfkl.Dense(self.n_data, name="dec-mu")(h)
        if n_sort > 0:
            x_mu = SplitSortLayer(n_sort, name="dec-mu-sortslice")(x_mu)

        # Log-Variance Head (compact + stable)
        x_logvar = tfkl.Dense(
            self.n_data,
            activation=lambda t: tf.math.log(self.min_variance + tf.nn.softplus(t)),
            name="dec-logvar"
        )(h)

        decoder = tfk.Model([z_in, cond_in], [x_mu, x_logvar], name="decoder")
        _ = decoder([tf.zeros((1, self.n_latent)), tf.zeros((1, self.n_conditions))])
        set_initialization(model=decoder, names=names, kernel_init=kernel_std, 
                           bias_init=bias_std)
        return decoder
