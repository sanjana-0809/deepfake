"""
fft_analysis.py — FFT-Based GAN / Diffusion Artifact Detection

Real photographs have a characteristic 1/f power spectrum with a smooth
radial falloff. GAN-generated and diffusion-model images often introduce
periodic peaks, grid patterns, or anomalous high-frequency energy that
deviates from this natural distribution.

This module:
  1. Converts the image to grayscale.
  2. Applies a 2-D FFT and computes the log-magnitude spectrum.
  3. Extracts a radial frequency profile.
  4. Scores the image by measuring deviations from the expected natural
     falloff using kurtosis, high-frequency energy ratio, and spectral
     flatness — combining them into a single fft_score ∈ [0, 1].

A score close to 1 suggests AI-generated / manipulated content.
A score close to 0 suggests a natural photograph.
"""

import numpy as np
from scipy.stats import kurtosis
import cv2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_magnitude_spectrum(gray: np.ndarray) -> np.ndarray:
    """
    Return the centered log-magnitude FFT spectrum of a grayscale image.

    Args:
        gray : 2-D uint8 grayscale image.

    Returns:
        spectrum : 2-D float32 log-magnitude, same shape as `gray`.
    """
    f = np.fft.fft2(gray.astype(np.float32))
    f_shifted = np.fft.fftshift(f)
    magnitude = np.abs(f_shifted)
    # Add 1 to avoid log(0)
    log_mag = np.log1p(magnitude)
    return log_mag.astype(np.float32)


def _radial_profile(spectrum: np.ndarray, num_bins: int = 64) -> np.ndarray:
    """
    Compute the mean magnitude along concentric rings (radial average).

    Returns a 1-D array of length `num_bins` representing how energy
    is distributed from the DC component (index 0) to the highest
    spatial frequency (index num_bins-1).
    """
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2

    y_idx, x_idx = np.indices((h, w))
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)

    max_r = np.sqrt(cx ** 2 + cy ** 2)
    bin_edges = np.linspace(0, max_r, num_bins + 1)

    profile = np.zeros(num_bins, dtype=np.float32)
    for i in range(num_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if mask.any():
            profile[i] = spectrum[mask].mean()

    return profile


# ---------------------------------------------------------------------------
# Scoring sub-components
# ---------------------------------------------------------------------------

def _high_frequency_energy_ratio(profile: np.ndarray,
                                  high_freq_fraction: float = 0.3) -> float:
    """
    Fraction of total energy concentrated in the top `high_freq_fraction`
    of spatial frequencies.

    Real images: low ratio (energy concentrated at low frequencies).
    GAN/Diffusion images: elevated ratio due to grid artifacts.
    """
    split = int(len(profile) * (1 - high_freq_fraction))
    low_energy = profile[:split].sum()
    high_energy = profile[split:].sum()
    total = low_energy + high_energy
    if total == 0:
        return 0.0
    return float(high_energy / total)


def _spectral_flatness(profile: np.ndarray) -> float:
    """
    Spectral flatness (Wiener entropy): geometric mean / arithmetic mean.

    Flat spectrum (close to 1) → noise-like or GAN-generated.
    Peaked spectrum (close to 0) → natural image with concentrated energy.

    We invert so that 1 = anomalous (flat / GAN-like).
    """
    profile = profile + 1e-8  # avoid log(0)
    geometric_mean = np.exp(np.mean(np.log(profile)))
    arithmetic_mean = np.mean(profile)
    flatness = geometric_mean / arithmetic_mean
    # Invert: natural images have low flatness; we want high score = suspicious
    return float(1.0 - np.clip(flatness, 0, 1))


def _kurtosis_score(profile: np.ndarray) -> float:
    """
    Excess kurtosis of the radial power profile.

    GAN images often have sharp spectral peaks → higher kurtosis.
    Natural images have smoother falloff → lower kurtosis.

    We map kurtosis to [0, 1]: higher kurtosis → higher score.
    """
    k = float(kurtosis(profile, fisher=True))
    # Map via sigmoid-like clamp: kurtosis typically in [-2, 10+] for images
    k_clipped = np.clip(k, -2, 12)
    score = (k_clipped + 2) / 14.0   # linear normalise to [0, 1]
    return float(np.clip(score, 0, 1))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_fft_score(image: np.ndarray) -> dict:
    """
    Compute the FFT-based manipulation score for an image.

    Args:
        image : uint8 RGB or BGR image of any size, shape (H, W, 3).

    Returns:
        dict with keys:
            fft_score          : float in [0, 1]  (1 = likely AI-generated)
            high_freq_ratio    : float in [0, 1]
            spectral_flatness  : float in [0, 1]
            kurtosis_score     : float in [0, 1]
            radial_profile     : np.ndarray (64,) for optional visualisation
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Resize to a fixed shape for consistent scoring
    gray = cv2.resize(gray, (256, 256))

    spectrum = _log_magnitude_spectrum(gray)
    profile = _radial_profile(spectrum, num_bins=64)

    hf_ratio = _high_frequency_energy_ratio(profile, high_freq_fraction=0.3)
    sf_score = _spectral_flatness(profile)
    kurt_score = _kurtosis_score(profile)

    # Weighted combination of the three sub-scores
    # High-frequency ratio carries the most discriminative power
    fft_score = (0.50 * hf_ratio) + (0.25 * sf_score) + (0.25 * kurt_score)
    fft_score = float(np.clip(fft_score, 0.0, 1.0))

    return {
        "fft_score": fft_score,
        "high_freq_ratio": float(hf_ratio),
        "spectral_flatness": float(sf_score),
        "kurtosis_score": float(kurt_score),
        "radial_profile": profile,
    }


def get_fft_spectrum_image(image: np.ndarray) -> np.ndarray:
    """
    Return a displayable uint8 RGB image of the FFT log-magnitude spectrum.
    Useful for visualisation in the Streamlit dashboard.

    Args:
        image : uint8 RGB image.

    Returns:
        spectrum_rgb : uint8 RGB image (256 x 256).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    gray = cv2.resize(gray, (256, 256))
    spectrum = _log_magnitude_spectrum(gray)

    # Normalise to [0, 255]
    spectrum_norm = cv2.normalize(spectrum, None, 0, 255,
                                  cv2.NORM_MINMAX).astype(np.uint8)
    spectrum_rgb = cv2.cvtColor(spectrum_norm, cv2.COLOR_GRAY2RGB)
    return spectrum_rgb
