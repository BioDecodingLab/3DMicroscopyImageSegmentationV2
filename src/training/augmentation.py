from typing import Any, Dict, List

import albumentations as A
import numpy as np
from scipy.ndimage import convolve, zoom
from scipy.signal import fftconvolve
import tifffile as tiff


class Augmentor:
    """Class for handling image augmentations."""

    def __init__(self, intensity_params: Dict[str, Any]):
        self.intensity_params = intensity_params

    @staticmethod
    def normalize_psf(psf: np.ndarray) -> np.ndarray:
        """Normalize PSF to sum to 1."""
        psf_sum = np.sum(psf)
        if not np.isclose(psf_sum, 1.0):
            psf = psf / psf_sum
        return psf

    def convolve_with_psf(self, image: np.ndarray, psf: np.ndarray) -> np.ndarray:
        """Convolve 3D image with PSF."""
        psf = self.normalize_psf(psf)
        pad_size = psf.shape
        padded_image = np.pad(image, [(p, p) for p in pad_size] + [(0, 0)], mode="reflect")
        convolved_padded = fftconvolve(padded_image, psf[..., np.newaxis], mode="same")
        slices = tuple(slice(p, -p) for p in pad_size) + (slice(None),)
        convolved_image = convolved_padded[slices]
        return np.clip(convolved_image, 0, None)

    def simulate_local_variations(
        self, shape: tuple, binnings: List[int] = None, scale: int = 5
    ) -> np.ndarray:
        """Simulate local variations in staining intensity."""
        if binnings is None:
            binnings = [1, 2, 4, 8]
        result = np.ones(shape)
        for binning in binnings:
            small_shape = tuple(dim // binning for dim in shape)
            zoom_factors = [
                orig_dim / small_dim for orig_dim, small_dim in zip(shape, small_shape)
            ]
            local_var = np.random.normal(size=small_shape) + 1.0
            smoothed = convolve(local_var, np.ones((scale, scale, scale)) / scale**3)
            variation = zoom(smoothed, zoom_factors, order=1)
            result *= variation
        return result

    def calculate_snr(
        self, signal: np.ndarray, noise: np.ndarray, mask: np.ndarray = None
    ) -> float:
        """Calculate the Signal-to-Noise Ratio (SNR) for a given signal and noise.

        Returns:
            SNR value or 0.0 if signal is empty or noise std is 0
        """
        if mask is not None:
            signal = signal * mask

        non_zero_values = signal[signal != 0]
        if len(non_zero_values) == 0:
            return 0.0

        signal_mean = np.mean(non_zero_values)
        noise_std = np.std(noise)

        if noise_std == 0 or np.isnan(noise_std):
            return 0.0

        return signal_mean / noise_std

    def generate_SNR_image(
        self,
        noisy_poisson_image: np.ndarray,
        convolved_image: np.ndarray,
        binary_mask: np.ndarray,
        snr_target: float,
        snr_tolerance: float,
        std_dev: float,
        max_iterations: int,
    ) -> np.ndarray:
        """Generate an image with a target Signal-to-Noise Ratio (SNR)."""
        iteration = 0
        snr = 0

        while iteration < max_iterations:
            final_image = noisy_poisson_image + np.random.normal(
                0, std_dev, noisy_poisson_image.shape
            )
            final_image = np.clip(final_image, 0, None)

            noise = final_image - convolved_image
            snr = self.calculate_snr(convolved_image, noise, binary_mask)

            if np.abs(snr - snr_target) < snr_tolerance or std_dev < 1e-6:
                return final_image

            if snr < snr_target:
                std_dev *= 0.9
            else:
                std_dev *= 1.1

            iteration += 1

        return final_image

    def augment_patch_intensity(self, patch: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Apply intensity-based augmentations to a 3D patch."""
        params = self.intensity_params

        if params["snr_targets"] and len(params["snr_targets"]) > 0:
            possible_snrs = params["snr_targets"] + [0] * (3 * len(params["snr_targets"]))
            target_snr = np.random.choice(possible_snrs)

            if target_snr == 0:
                return patch
            else:
                if mask is not None:
                    mask = mask.astype(np.uint16)
                    mask[mask > 0] = 1
                    patch = mask * params["intensity_scale"]

                if 0 < params["z_decay_rate"] < 1:
                    z_profile = np.exp(-np.arange(patch.shape[0]) * (1 - params["z_decay_rate"]))
                    patch = patch * z_profile[:, np.newaxis, np.newaxis, np.newaxis]

                local_var = self.simulate_local_variations(
                    patch.shape[:-1],
                    binnings=[1, 2, 4, 8],
                    scale=params["local_variation_scale"],
                )
                patch = patch * local_var[..., np.newaxis]
                patch = np.clip(patch, 0, None)

                if params["background_level"] > 0:
                    bg = (
                        np.ones_like(patch)
                        * params["background_level"]
                        * params["intensity_scale"]
                    )
                    bg_noise = (
                        bg
                        * self.simulate_local_variations(
                            patch.shape[:-1],
                            binnings=[1, 2],
                            scale=params["local_variation_scale"],
                        )[..., np.newaxis]
                    )
                    bg_noise[patch != 0] = 0
                    bg_noise = np.clip(bg_noise, 0, None)
                    patch = patch + bg_noise

                convolved_image = None
                if params.get("use_psf") and params.get("psf_path"):
                    psf = tiff.imread(params["psf_path"])
                    convolved_image = self.convolve_with_psf(patch, psf)
                    patch = convolved_image

                if params["poisson_scale"] > 0:
                    scaled = patch * params["poisson_scale"]
                    patch = np.random.poisson(scaled) / params["poisson_scale"]
                    patch = np.clip(patch, 0, None)

                if convolved_image is not None:
                    patch = self.generate_SNR_image(
                        patch,
                        convolved_image,
                        mask,
                        target_snr,
                        params["snr_tolerance"],
                        params["std_dev"],
                        params["max_iterations"],
                    )

                return patch
        else:
            return patch

    @staticmethod
    def create_standard_augmentation_pipeline() -> A.Compose:
        """Create an augmentation pipeline for 3D patches."""
        return A.Compose(
            [
                A.CubicSymmetry(p=0.7),
                A.CoarseDropout3D(p=0.3),
            ]
        )
