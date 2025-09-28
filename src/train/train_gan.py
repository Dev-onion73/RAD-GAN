# src/train/train_gan.py
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import optimizers, losses
from src.models.gan_models import build_generator, build_patch_discriminator

class GANTrainer:
    def __init__(self, img_shape=(64,64,1), lr=2e-4, gp_weight=10.0, recon_weight=50.0):
        self.gen = build_generator(input_shape=img_shape)
        self.disc = build_patch_discriminator(input_shape=img_shape)
        self.gen_optimizer = optimizers.Adam(lr, beta_1=0.5)
        self.disc_optimizer = optimizers.Adam(lr, beta_1=0.5)
        self.recon_weight = recon_weight
        self.loss_bce = losses.BinaryCrossentropy(from_logits=True)
        # For gradient penalty if needed, implement WGAN-GP optionally.

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        # noise as same shape as image (map) - easier conditioning
        noise = tf.random.normal(shape=tf.shape(real_images))
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.gen(noise, training=True)
            # Discriminator logits (Patch)
            real_logits = self.disc(real_images, training=True)
            fake_logits = self.disc(fake_images, training=True)
            # Adversarial loss (hinge-style)
            d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - real_logits))
            d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + fake_logits))
            d_loss = d_loss_real + d_loss_fake
            # Generator adv term
            g_adv = -tf.reduce_mean(fake_logits)
            # Reconstruction (L1) to stabilize conditional behaviour
            g_recon = tf.reduce_mean(tf.abs(fake_images - real_images))
            g_loss = g_adv + self.recon_weight * g_recon

        # gradients
        disc_grads = disc_tape.gradient(d_loss, self.disc.trainable_variables)
        gen_grads = gen_tape.gradient(g_loss, self.gen.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(disc_grads, self.disc.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.gen.trainable_variables))
        return {'d_loss': d_loss, 'g_loss': g_loss, 'g_recon': g_recon}

def train_dataset(specs, batch_size=16, epochs=20, img_shape=(64,64,1), out_dir='runs'):
    os.makedirs(out_dir, exist_ok=True)
    trainer = GANTrainer(img_shape)
    dataset = tf.data.Dataset.from_tensor_slices(specs).shuffle(1000).batch(batch_size).prefetch(2)
    for epoch in range(epochs):
        logs = {'d_loss':[], 'g_loss':[], 'g_recon':[]}
        for batch in dataset:
            # ensure shape (B,H,W,1)
            if batch.ndim==3:
                batch = tf.expand_dims(batch, -1)
            metrics = trainer.train_step(batch)
            for k,v in metrics.items():
                logs[k].append(float(tf.reduce_mean(v)))
        print(f"Epoch {epoch+1}/{epochs}  d_loss={np.mean(logs['d_loss']):.4f} g_loss={np.mean(logs['g_loss']):.4f} g_recon={np.mean(logs['g_recon']):.4f}")
        # save sample and checkpoint
        if (epoch+1) % 5 == 0:
            ckpt_path = os.path.join(out_dir, f'gen_epoch_{epoch+1}.h5')
            trainer.gen.save(ckpt_path)
    return trainer
