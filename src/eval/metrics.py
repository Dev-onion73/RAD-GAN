# src/eval/metrics.py
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.linalg import sqrtm

def compute_ssim_batch(real, fake):
    # real/fake: (N, F, T) or (N, F, T, 1)
    if real.ndim==4: real = np.squeeze(real, -1)
    if fake.ndim==4: fake = np.squeeze(fake, -1)
    vals = []
    for r, f in zip(real, fake):
        vals.append(ssim(r, f, data_range=1.0))
    return float(np.mean(vals))

def doppler_centroid(spectrogram):
    F = spectrogram.shape[0]
    freqs = np.arange(F)
    # centroids over time
    cent = np.sum(spectrogram * freqs[:, None], axis=0) / (np.sum(spectrogram, axis=0) + 1e-9)
    return cent

def centroid_rmse(real_batch, fake_batch):
    if real_batch.ndim==4: real_batch = np.squeeze(real_batch, -1)
    if fake_batch.ndim==4: fake_batch = np.squeeze(fake_batch, -1)
    errs = []
    for r, f in zip(real_batch, fake_batch):
        cr = doppler_centroid(r)
        cf = doppler_centroid(f)
        errs.append(np.sqrt(np.mean((cr-cf)**2)))
    return float(np.mean(errs))

# Fréchet Radar Distance (FRD) using features from an encoder
def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = sqrtm((sigma1 + eps*np.eye(sigma1.shape[0])).dot(sigma2 + eps*np.eye(sigma2.shape[0])))
    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean))

def compute_frd(feature_extractor, real_images, fake_images, batch=64):
    """
    feature_extractor: a model mapping images to d-dim features (numpy input/output)
    real_images/fake_images: arrays (N,H,W,1)
    """
    # get features
    feats_r = feature_extractor.predict(real_images, batch_size=batch)
    feats_g = feature_extractor.predict(fake_images, batch_size=batch)
    mu_r = np.mean(feats_r, axis=0); mu_g = np.mean(feats_g, axis=0)
    sigma_r = np.cov(feats_r, rowvar=False); sigma_g = np.cov(feats_g, rowvar=False)
    return frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
