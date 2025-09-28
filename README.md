---

## 🧩 Pipeline Overview

Think of the project in **4 stages**:

1. **Simulation (data generation)**

   * We don’t have radar hardware → so we *simulate* signals and transform them into spectrogram “images.”
   * These spectrograms are the training data for our GAN.

2. **Modeling (GAN setup)**

   * We build a **generator** (to create fake spectrograms) and a **discriminator** (to judge them).
   * Training pits them against each other until the generator learns to produce realistic-looking spectrograms.

3. **Training loop**

   * Feeds real spectrograms + noise to the GAN.
   * Updates both generator & discriminator weights.
   * Saves checkpoints so we can later generate new samples without retraining.

4. **Evaluation & Visualization**

   * Compute similarity metrics (SSIM, Doppler centroid RMSE, FRD).
   * Plot side-by-side comparisons of real vs generated spectrograms.
   * Judge how well the GAN has learned.

---

## 🔍 What Each Code File Does

### **1. Simulator (`src/sim/microdoppler.py`)**

* Generates synthetic micro-Doppler signals:

  * A rotating scatterer produces sinusoidal radial velocities → converted to a baseband time series.
  * Then we compute its **spectrogram** (time–frequency image).
* Output = normalized spectrogram (`float32` array).
  👉 This gives us *fake “radar returns”* without needing hardware.

---

### **2. Data I/O (`src/data/io.py`)**

* Utility to save/load datasets in `.npy` format.
* Keeps both the **raw spectrogram arrays** and optional **metadata** (like simulated rotation speed).
  👉 Handles dataset persistence — so you can generate once, reuse many times.

---

### **3. GAN Models (`src/models/gan_models.py`)**

* **Generator:** UNet-style → takes random noise (or conditioned noise) and outputs a spectrogram-shaped image.
* **Discriminator:** PatchGAN → classifies whether *local patches* of an image are real or fake.
  👉 Together, they form the adversarial game: generator tries to fool discriminator, discriminator tries to spot fakes.

---

### **4. Training Engine (`src/train/train_gan.py`)**

* Wraps the GAN in a `GANTrainer` class.
* Implements a custom `train_step` using `tf.GradientTape`:

  * Update discriminator with real vs fake spectrograms.
  * Update generator to minimize adversarial loss + reconstruction (L1) loss.
* `train_dataset` drives the full loop over many epochs, printing losses and saving checkpoints.
  👉 This is the heart of learning — turning simulated data into a trained model.

---

### **5. Metrics (`src/eval/metrics.py`)**

* **SSIM**: Structural similarity (perceptual image similarity).
* **Centroid RMSE**: Compare Doppler centroids (physics-relevant feature).
* **FRD**: Fréchet Radar Distance (like FID, but for spectrograms, using a learned encoder).
  👉 These tell us not just if generated spectrograms look good, but if they *match radar-like features*.

---

### **6. Encoder (`src/models/encoders.py`)**

* Small CNN that produces feature vectors for spectrograms.
* Used inside FRD calculation to get embeddings for real vs fake distributions.
  👉 Gives a more meaningful measure of distribution overlap than pixel-wise comparison.

---

### **7. Visualization (`src/viz/viz.py`)**

* Simple matplotlib helpers:

  * Show one spectrogram.
  * Compare real vs fake grids.
    👉 Makes it easy to visually verify if generator outputs “look right.”

---

### **8. Scripts**

* **`scripts/gen_dataset.py`**
  Generates a dataset of N synthetic spectrograms → saves as `.npy`.

* **`scripts/demo_run.py`**
  End-to-end mini demo:

  1. Load dataset.
  2. Train GAN for a few epochs.
  3. Generate fake samples.
  4. Compute SSIM, RMSE, FRD.
  5. Print results.
     👉 This is the “run once and see everything working” script.

---

## 🔄 Full Pipeline Flow

1. **Dataset Creation**
   `gen_dataset.py` → calls `simulate_and_spectrogram` → saves dataset.

2. **Training**
   `demo_run.py` → loads dataset → `train_dataset` trains GAN → saves generator model.

3. **Generation**
   After training, generator produces fake spectrograms from noise.

4. **Evaluation**
   Metrics (`SSIM`, `RMSE`, `FRD`) + visualization grid to assess quality.

5. **Iteration**

   * If poor quality → tweak model (layers, learning rate) or data (bigger dataset).
   * If good → you now have a spectrogram simulator that “hallucinates” plausible radar micro-Doppler patterns.

---

✅ That’s the end-to-end:

* **Simulate → Train → Generate → Evaluate → Visualize.**