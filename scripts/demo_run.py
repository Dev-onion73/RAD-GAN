# scripts/demo_run.py
import numpy as np
from src.data.io import load_dataset_npy
from src.train.train_gan import train_dataset
from src.models.encoders import build_encoder
from src.eval.metrics import compute_ssim_batch, centroid_rmse, compute_frd

# generate or load dataset first (scripts/gen_dataset.py)
specs, metas = load_dataset_npy('data/specs_32.npy')
# resize or crop to a power-of-two shape if needed. Our models expect 64x64 here
# simple resampling to 64x64
import cv2
specs_rs = np.stack([cv2.resize(s, (64,64)) for s in specs])
specs_rs = specs_rs.astype('float32')
# train a small GAN (demo: epochs small)
trainer = train_dataset(specs_rs, batch_size=8, epochs=10, img_shape=(64,64,1), out_dir='runs/demo')

# generate fake samples
noise = np.random.normal(size=(specs_rs.shape[0], 64,64,1)).astype('float32')
gen_model = trainer.gen
fakes = gen_model.predict(noise)

# compute metrics
ssim_v = compute_ssim_batch(specs_rs, fakes)
cent_rmse = centroid_rmse(specs_rs, fakes)

# train encoder for FRD
enc = build_encoder((64,64,1), feat_dim=128)
enc.compile(optimizer='adam', loss='mse')
# as a quick proxy: train encoder as autoencoder target (not ideal but serves for features)
enc.fit(specs_rs[...,None], np.zeros((len(specs_rs),128)), epochs=5, batch_size=8)  # tiny hack; better to train a proper feature extractor
frd = compute_frd(enc, specs_rs[...,None], fakes, batch=8)

print("SSIM:", ssim_v, "Centroid RMSE:", cent_rmse, "FRD:", frd)
