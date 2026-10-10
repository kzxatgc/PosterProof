"""
PosterProof — Evidence-First Event Poster Audit.
Compare a printed poster photo against an approved master to surface evidence of changes.
"""

import base64
import cgi
import html
import io
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np


class AuditError(Exception):
    """Raised when audit input is invalid or evidence is insufficient to compare."""
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum accepted upload size per image (20 MB; suits local event-staff use).
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def decode_image(data: bytes, label: str) -> np.ndarray:
    """Decode raw image bytes into a BGR array; raise AuditError on failure."""
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise AuditError(f"{label}: missing image or upload exceeds the size limit.")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise AuditError(f"{label}: unsupported or damaged image file.")
    return image


def crop_roi(image: np.ndarray, roi: tuple, label: str) -> np.ndarray:
    """Slice a rectangle from *image*; raise AuditError if the rect is out of bounds."""
    x, y, w, h = roi
    ih, iw = image.shape[:2]
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > iw or y + h > ih:
        raise AuditError(f"{label} rectangle is outside the image.")
    return image[y:y + h, x:x + w].copy()


def _roi_from_fields(fields: dict, prefix: str) -> tuple:
    """Parse x/y/w/h integer fields from a form-field dict with a given prefix."""
    try:
        x = int(fields[f"{prefix}_x"])
        y = int(fields[f"{prefix}_y"])
        w = int(fields[f"{prefix}_w"])
        h = int(fields[f"{prefix}_h"])
    except (KeyError, ValueError) as exc:
        raise AuditError(f"Invalid or missing ROI fields for '{prefix}': {exc}") from exc
    return (x, y, w, h)


def decode_qr(cropped_roi: np.ndarray) -> str:
    """Decode a QR code from a pre-cropped ROI array only; never searches the whole image."""
    text, _points, _straight = cv2.QRCodeDetector().detectAndDecode(cropped_roi)
    return text or ""


def _to_data_uri(image: np.ndarray) -> str:
    """Encode a BGR image array to a base64 PNG data URI (in-memory; no temp files)."""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise AuditError("Failed to encode image for display.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Image Registration
# ---------------------------------------------------------------------------

def register_image(reference: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Align observed photo to reference poster geometry using ORB and homography."""
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    obs_gray = cv2.cvtColor(observed, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp_reference, desc_reference = orb.detectAndCompute(ref_gray, None)
    kp_observed, desc_observed = orb.detectAndCompute(obs_gray, None)

    if desc_reference is None or desc_observed is None or len(kp_reference) < 12 or len(kp_observed) < 12:
        raise AuditError("Could not compare reliably: insufficient visual features detected.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = matcher.knnMatch(desc_observed, desc_reference, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]

    if len(good) < 12:
        raise AuditError("Could not compare reliably: too few feature matches.")

    src_points = np.float32([kp_observed[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_points = np.float32([kp_reference[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 4.0)
    if H is None or mask is None or int(mask.sum()) < 12:
        raise AuditError("Could not compare reliably: registration evidence is weak.")

    h_ref, w_ref = reference.shape[:2]
    registered = cv2.warpPerspective(observed, H, (w_ref, h_ref))
    return registered

