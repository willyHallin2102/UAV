from __future__ import annotations

import tensorflow as tf
tfk, tfkl = tf.keras, tf.keras.layers
from tensorflow.keras.optimizers import Adam

from typing import List, Sequence
from src.models.utilities.helpers import (
    set_initialization, extract_inputs,
    SplitSortLayer
)

from src.math.random import generate_noise



class WGan_GP(tfk.Model):
    """
        Wasserstein (W) Generative Adversarial Network (GAN)
        Gradient Penalty (GP).

        This model trains a generator and a critic to approximate 
        the wasserstein distance between real and generated 
        distributions. Gradient Penalty enforces the 
        Lipscitz constraints
    """
    def __init__(self,
        n_latent: int, n_data: int, n_conditions: int,
        generator_layers    : Sequence[int] = (256, 128),
        critic_layers       : Sequence[int] = (128, 256),
        gp_lambda           : float = 10.0,
        n_critical          : int   = 5,
        init_kernel         : float = 10.0,
        init_bias           : float = 10.0,
        n_sort              : int   = 0,
        **kwargs 
    ):
        """
        """
        super().__init__(**kwargs)

        self.n_latent, self.n_data = int(n_latent), int(n_data)
        self.n_conditions = int(n_conditions)

        self.gp_lambda = float(gp_lambda)
        self.n_critic = int(n_critical)

        # Build generator & critic
        self.generator = self._build_generator()
    

    # --------------------------- Metrics --------------------------- #

    @property
    def metrics(self) -> List[tfk.metrics.Metric]:
        """
        """
        return [self.generator_loss_tracker,
                self.critic_loss_tracker,
                self.wasserstein_dist,
                self.gradient_penalty_tracker]
    

    def _initialize_metrics(self):
        """
        """
        self.generator_loss_tracker = tfk.metrics.Mean(name="generator-loss")
        self.critic_loss_tracker = tfk.metrics.Mean(name="critical-loss")
        self.wasserstein_dist = tfk.metrics.Mean(name="wasserstein-dist")
        self.gradient_penalty_tracker = tfk.metrics.Mean(name="gradient-penalty")
    

    def compile(self,
        g_opt: tfk.optimizers.Optimizer = Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9),
        c_opt: tfk.optimizers.Optimizer = Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9),
        **kwargs
    ):
        """
        """
        super().compile(**kwargs)
        self.g_opt, self.c_opt = g_opt, c_opt
    

    # --------------------------- Forward --------------------------- #

    def call(self, inputs, training: bool=False) -> tf.Tensor:
        """
        """
        z, conditions = extract_inputs(inputs)
        return self.generator([z, conditions], training=training)
    

    # --------------------------- Training --------------------------- #

    def train_step(self, inputs):
        """
            One training step with n_critics critic updates
        """
        x_real, conditions = extract_inputs(inputs)
        batch_size = tf.shape(x_real)[0]

        # Train critic multiple times
        for _ in tf.range(self.n_critic):
            z = generate_noise(batch_size, self.n_latent, backend="tf")
            with tf.GradientTape() as critic_tape:
                x_fake = self.generator([z, conditions], training=True)
                real_scores = self.critic([x_real, conditions], training=True)
                fake_scores = self.critic([x_fake, conditions], training=True)

                wasserstein_loss = tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)
                gradient_penalty = self._gradient_penalty(x_real, x_fake, conditions)
                critic_loss = wasserstein_loss +self.gp_lambda * gradient_penalty
            
            critic_gradients = critic_tape.gradient(critic_loss, 
                                                    self.critic.trainable_variables)
            self.c_opt.apply_gradients(zip(critic_gradients, 
                                           self.critic.trainable_variables))
        # Train the Generator
        z = generate_noise(batch_size, self.n_latent, backend="tf")
        with tf.GradientTape() as generator_tape:
            x_fake = self.generator([z, conditions], training=True)
            fake_scores = self.critic([x_fake, conditions], training=True)
            generator_loss = -tf.reduce_mean(fake_scores)
        
        generator_gradients = generator_tape.gradient(generator_loss,
                                                      self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(generator_gradients, 
                                       self.generator.trainable_variables))
        # Update Metrics
        self.generator_loss_tracker.update_state(generator_loss)
        self.critic_loss_tracker.update_state(critic_loss)
        self.wasserstein_dist.update_state(wasserstein_loss)
        self.gradient_penalty_tracker.update_state(gradient_penalty)

        return {metric.name: metric.result() for metric in self.metrics}
    

    def test_step(self, inputs):
        """
        """
        x_real, conditions = extract_inputs(inputs)
        batch_size = tf.shape(x_real)[0]
        z = generate_noise(batch_size, self.n_latent, backend="tf")

        x_fake = self.generator([z, conditions], training=False)
        real_scores = self.critic([x_real, conditions], training=False)
        fake_scores = self.critic([x_fake, conditions], training=False)

        wasserstein_loss = tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)
        gradient_penalty = self._gradient_penalty(x_real, x_fake, conditions)
        critic_loss = wasserstein_loss + self.gp_lambda * gradient_penalty
        generator_loss = -tf.reduce_mean(fake_scores)

        self.generator_loss_tracker.update_state(generator_loss)
        self.critic_loss_tracker.update_state(critic_loss)
        self.wasserstein_dist.update_state(wasserstein_loss)
        self.gradient_penalty_tracker.update_state(gradient_penalty)

        return {metric.name: metric.result() for metric in self.metrics}
    
    # --------------------------- Utilities --------------------------- #

    def generate_samples(self, n: int, cond: tf.Tensor=None) -> tf.Tensor:
        """Generate samples for evaluation/visualization."""
        z = generate_noise(n, self.n_latent, backend="tf")
        if cond is None: cond = tf.zeros((n, self.n_conditions))
        return self.generator([z, cond], training=False)
    


    # --------------------------- Internals --------------------------- #

    def _gradient_penalty(self, real: tf.Tensor, fake: tf.Tensor, cond: tf.Tensor) -> tf.Tensor:
        """Gradient penalty enforcing 1-Lipschitz constraint."""
        batch_size = tf.shape(real)[0]
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        alpha = tf.broadcast_to(alpha, tf.shape(real))

        interpolated = real + alpha * (fake - real)

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            scores = self.critic([interpolated, cond], training=True)
        grads = tape.gradient(scores, interpolated)

        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]) + 1e-12)
        return tf.reduce_mean(tf.square(grad_norm - 1.0))


    def _build_generator(self, layers: Sequence[int], kernel_std: float, bias_std: float, n_sort: int) -> tfk.Model:
        z_in = tfkl.Input(shape=(self.n_latent,), name="gen_z")
        cond_in = tfkl.Input(shape=(self.n_conditions,), name="gen_cond")
        h = tfkl.Concatenate()([z_in, cond_in])

        names: List[str] = []
        for i, units in enumerate(layers):
            name = f"gen_dense_{i}"
            h = tfkl.Dense(units, activation="relu", name=name)(h)
            names.append(name)

        x_out = tfkl.Dense(self.n_data, activation="linear", name="gen_out")(h)

        if n_sort > 0:
            x_sort = SplitSortLayer(n_sort=n_sort)(x_out[:, :n_sort])
            x_rest = x_out[:, n_sort:]
            x_out = tfkl.Concatenate(axis=1)([x_sort, x_rest])

        model = tfk.Model([z_in, cond_in], x_out, name="generator")

        _ = model([tf.zeros((1, self.n_latent)), tf.zeros((1, self.n_conditions))])
        set_initialization(model, names, kernel_std, bias_std)
        return model


    def _build_critic(self, layers: Sequence[int], kernel_std: float, bias_std: float) -> tfk.Model:
        x_in = tfkl.Input(shape=(self.n_data,), name="crit_x")
        cond_in = tfkl.Input(shape=(self.n_conditions,), name="crit_cond")
        h = tfkl.Concatenate()([x_in, cond_in])

        names: List[str] = []
        for i, units in enumerate(layers):
            name = f"crit_dense_{i}"
            h = tfkl.Dense(units, name=name)(h)
            h = tfkl.LeakyReLU(0.2)(h)
            names.append(name)

        scores = tfkl.Dense(1, name="crit_out")(h)
        model = tfk.Model([x_in, cond_in], scores, name="critic")

        _ = model([tf.zeros((1, self.n_data)), tf.zeros((1, self.n_conditions))])
        set_initialization(model, names, kernel_std, bias_std)
        return model
