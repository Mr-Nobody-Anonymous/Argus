"""
Image enhancement algorithms for improved detection in low-light, noisy, or blurry conditions
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple
from backend.config.config import get_config, section_to_dict

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


class ImageEnhancement:
    """
    Provides image enhancement algorithms:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization) for low-light
    - Denoising (Non-local Means)
    - Super-resolution via edge enhancement
    - Night vision enhancement (infrared-style)
    - Deblurring (Wiener filter)
    - Auto-white balance correction
    """

    def __init__(self):
        self.config = get_config()
        self.enhancement_config = section_to_dict(getattr(self.config, 'enhancement', {}))
        self.enabled = self.enhancement_config.get('enabled', True)
        if cv2 is None or np is None:
            self.enabled = False
            logger.warning("Image enhancement dependencies unavailable, enhancement disabled")

    def enhance_frame(self, frame: np.ndarray, mode: str = "auto") -> np.ndarray:
        """
        Enhance a video frame using the specified mode.
        
        Modes:
            - "auto": Automatically detect and apply best enhancement
            - "low_light": CLAHE + brightness boost
            - "denoise": Remove noise while preserving edges
            - "sharpen": Edge enhancement for clearer details
            - "night": Night vision style enhancement
            - "deblur": Reduce motion blur
            - "hdr": HDR-like tone mapping
        """
        if not self.enabled:
            return frame

        if cv2 is None or np is None:
            return frame

        try:
            if mode == "auto":
                return self._auto_enhance(frame)
            elif mode == "low_light":
                return self._enhance_low_light(frame)
            elif mode == "denoise":
                return self._denoise_frame(frame)
            elif mode == "sharpen":
                return self._sharpen_frame(frame)
            elif mode == "night":
                return self._night_vision_enhance(frame)
            elif mode == "deblur":
                return self._deblur_frame(frame)
            elif mode == "hdr":
                return self._hdr_tone_map(frame)
            else:
                return frame
        except Exception as e:
            logger.error(f"Error enhancing frame: {e}")
            return frame

    def _auto_enhance(self, frame: np.ndarray) -> np.ndarray:
        """Automatically detect conditions and apply best enhancement"""
        # Calculate brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Calculate contrast (standard deviation)
        contrast = np.std(gray)
        
        # Calculate blurriness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        enhanced = frame.copy()
        
        # Low light condition
        if mean_brightness < 80:
            logger.debug(f"Low light detected (brightness={mean_brightness:.1f}), applying CLAHE")
            enhanced = self._enhance_low_light(enhanced)
        
        # Low contrast
        if contrast < 40:
            logger.debug(f"Low contrast detected (contrast={contrast:.1f}), applying contrast stretch")
            enhanced = self._contrast_stretch(enhanced)
        
        # Blurry image
        if laplacian_var < 100:
            logger.debug(f"Blurry image detected (laplacian={laplacian_var:.1f}), applying sharpen")
            enhanced = self._sharpen_frame(enhanced)
        
        # Always apply mild denoising
        enhanced = self._mild_denoise(enhanced)
        
        return enhanced

    def _enhance_low_light(self, frame: np.ndarray) -> np.ndarray:
        """CLAHE + brightness enhancement for low-light conditions"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # Merge channels
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Additional gamma correction for very dark areas
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if np.mean(gray) < 50:
                gamma = 1.5
                look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
                enhanced = cv2.LUT(enhanced, look_up_table)
            
            return enhanced
        except Exception as e:
            logger.error(f"Error in low light enhancement: {e}")
            return frame

    def _denoise_frame(self, frame: np.ndarray) -> np.ndarray:
        """Non-local means denoising (strong)"""
        try:
            return cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        except Exception as e:
            logger.error(f"Error denoising frame: {e}")
            return frame

    def _mild_denoise(self, frame: np.ndarray) -> np.ndarray:
        """Mild denoising for general use"""
        try:
            return cv2.fastNlMeansDenoisingColored(frame, None, 3, 3, 5, 15)
        except Exception as e:
            return frame

    def _sharpen_frame(self, frame: np.ndarray) -> np.ndarray:
        """Edge enhancement using unsharp masking"""
        try:
            # Gaussian blur
            blurred = cv2.GaussianBlur(frame, (0, 0), 3.0)
            
            # Unsharp masking
            sharpened = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
            
            # Additional kernel-based sharpening
            kernel = np.array([
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
            ])
            sharpened = cv2.filter2D(sharpened, -1, kernel)
            
            return sharpened
        except Exception as e:
            logger.error(f"Error sharpening frame: {e}")
            return frame

    def _night_vision_enhance(self, frame: np.ndarray) -> np.ndarray:
        """Night vision style enhancement (green-tinted, high contrast)"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            
            # Apply mild Gaussian blur to reduce noise
            enhanced_gray = cv2.GaussianBlur(enhanced_gray, (3, 3), 0.5)
            
            # Create green-tinted night vision effect
            night_vision = cv2.merge([
                np.zeros_like(enhanced_gray),  # Blue channel (minimal)
                enhanced_gray,                   # Green channel (full)
                np.zeros_like(enhanced_gray)     # Red channel (minimal)
            ])
            
            return night_vision
        except Exception as e:
            logger.error(f"Error in night vision enhancement: {e}")
            return frame

    def _deblur_frame(self, frame: np.ndarray) -> np.ndarray:
        """Reduce motion blur using Wiener filter approximation"""
        try:
            # Estimate blur kernel size from Laplacian
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Adjust deblurring strength based on blur amount
            if laplacian_var < 50:
                # Heavy blur - use stronger deblurring
                kernel_size = 5
                strength = 1.5
            elif laplacian_var < 100:
                kernel_size = 3
                strength = 1.2
            else:
                # Minimal blur - skip deblurring
                return frame
            
            # Apply deconvolution using Wiener filter approximation
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            
            deblurred = cv2.filter2D(frame, -1, kernel)
            deblurred = cv2.addWeighted(frame, strength, deblurred, 1 - strength, 0)
            
            return deblurred
        except Exception as e:
            logger.error(f"Error deblurring frame: {e}")
            return frame

    def _hdr_tone_map(self, frame: np.ndarray) -> np.ndarray:
        """HDR-like tone mapping for better dynamic range"""
        try:
            # Convert to float
            hdr = frame.astype(np.float32) / 255.0
            
            # Apply Reinhard tone mapping
            tonemap = cv2.createTonemapReinhard(gamma=2.2, intensity=0.0, light_adapt=0.8, color_adapt=0.6)
            hdr_result = tonemap.process(hdr)
            
            # Convert back to 8-bit
            result = (hdr_result * 255).astype(np.uint8)
            
            return result
        except Exception as e:
            logger.error(f"Error in HDR tone mapping: {e}")
            return frame

    def _contrast_stretch(self, frame: np.ndarray) -> np.ndarray:
        """Apply contrast stretching"""
        try:
            # Convert to YUV
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv)
            
            # Apply contrast stretching to Y channel
            min_val = np.percentile(y, 5)
            max_val = np.percentile(y, 95)
            
            if max_val > min_val:
                y_stretched = np.clip((y.astype(np.float32) - min_val) * 255.0 / (max_val - min_val), 0, 255).astype(np.uint8)
            else:
                y_stretched = y
            
            # Merge and convert back
            enhanced_yuv = cv2.merge([y_stretched, u, v])
            enhanced = cv2.cvtColor(enhanced_yuv, cv2.COLOR_YUV2BGR)
            
            return enhanced
        except Exception as e:
            logger.error(f"Error in contrast stretch: {e}")
            return frame

    def detect_quality_issues(self, frame: np.ndarray) -> Dict:
        """Detect image quality issues and return diagnostics"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Brightness
            mean_brightness = float(np.mean(gray))
            
            # Contrast
            contrast = float(np.std(gray))
            
            # Blurriness
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            
            # Noise estimation
            noise_estimate = float(np.std(cv2.GaussianBlur(gray, (5, 5), 0) - gray))
            
            issues = []
            if mean_brightness < 60:
                issues.append("very_dark")
            elif mean_brightness < 80:
                issues.append("low_light")
            if contrast < 30:
                issues.append("low_contrast")
            if laplacian_var < 50:
                issues.append("very_blurry")
            elif laplacian_var < 100:
                issues.append("blurry")
            if noise_estimate > 20:
                issues.append("noisy")
            
            return {
                "brightness": round(mean_brightness, 1),
                "contrast": round(contrast, 1),
                "sharpness": round(laplacian_var, 1),
                "noise_level": round(noise_estimate, 1),
                "issues": issues,
                "quality_score": round(self._calculate_quality_score(mean_brightness, contrast, laplacian_var, noise_estimate), 1)
            }
        except Exception as e:
            logger.error(f"Error detecting quality issues: {e}")
            return {"issues": ["error"], "quality_score": 0.0}

    def _calculate_quality_score(self, brightness: float, contrast: float, sharpness: float, noise: float) -> float:
        """Calculate overall image quality score (0-100)"""
        score = 100.0
        
        # Penalize extreme brightness
        if brightness < 40:
            score -= 30
        elif brightness < 70:
            score -= 15
        elif brightness > 220:
            score -= 20
        
        # Penalize low contrast
        if contrast < 20:
            score -= 25
        elif contrast < 40:
            score -= 10
        
        # Penalize blur
        if sharpness < 30:
            score -= 30
        elif sharpness < 60:
            score -= 15
        elif sharpness < 100:
            score -= 5
        
        # Penalize noise
        if noise > 30:
            score -= 20
        elif noise > 20:
            score -= 10
        
        return max(0, min(100, score))


# Global instance
_image_enhancement = None


def get_image_enhancement() -> ImageEnhancement:
    """Get global image enhancement instance"""
    global _image_enhancement
    if _image_enhancement is None:
        _image_enhancement = ImageEnhancement()
    return _image_enhancement