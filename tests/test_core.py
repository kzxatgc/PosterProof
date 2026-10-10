"""Core tests for PosterProof application."""

import io
import cv2
import numpy as np
import PIL.Image
import pytest
import qrcode

from app import (
    DEFAULT_EXPECTED_PAYLOAD,
    DEFAULT_FIELD_ROI,
    DEFAULT_QR_ROI,
    AuditError,
    audit_images,
    crop_roi,
    decode_image,
    decode_qr,
    register_image,
)
from tools.generate_samples import QR_BAD, QR_PAYLOAD, make_poster, perspective_photo


def _pil_to_png_bytes(im: PIL.Image.Image) -> bytes:
    """Convert a PIL Image into PNG byte stream."""
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def test_decode_image_rejects_junk_bytes():
    """Verify decode_image raises AuditError on un-decodable bytes."""
    with pytest.raises(AuditError, match="unsupported or damaged image file"):
        decode_image(b"not a valid image byte stream", "Master image")


def test_crop_roi_simple_rectangle():
    """Verify crop_roi slices the exact ROI rectangle."""
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    cropped = crop_roi(canvas, (10, 20, 30, 40), "Test ROI")
    assert cropped.shape == (40, 30, 3)


def test_decode_qr_reads_simple_qr():
    """Verify decode_qr extracts payload from a generated QR ROI."""
    qr = qrcode.QRCode(box_size=5, border=2)
    qr.add_data("POSTERPROOF_TEST")
    qr.make(fit=True)
    qr_pil = qr.make_image(fill_color="black", back_color="white")
    qr_bgr = cv2.cvtColor(np.array(qr_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    decoded = decode_qr(qr_bgr)
    assert decoded == "POSTERPROOF_TEST"


def test_feature_register_refuses_low_feature_count():
    """Verify register_image raises AuditError when images lack features."""
    ref = np.full((200, 200, 3), 255, dtype=np.uint8)
    obs = np.full((200, 200, 3), 255, dtype=np.uint8)
    with pytest.raises(AuditError, match="insufficient visual features"):
        register_image(ref, obs)


def test_audit_images_unchanged_scenario_pipeline():
    """Verify audit_images reports matched QR and unchanged field on identical posters."""
    master_im = make_poster()
    master_bytes = _pil_to_png_bytes(master_im)
    res = audit_images(master_bytes, master_bytes)

    assert res["qr_state"] == "QR PAYLOAD MATCHED"
    assert res["field_state"] == "NO THRESHOLD-EXCEEDING CHANGE FLAGGED IN MONITORED REGION"


def test_audit_images_payload_mismatch():
    """Verify audit_images reports MISMATCH DETECTED when QR payloads differ."""
    master_bytes = _pil_to_png_bytes(make_poster(payload=QR_PAYLOAD))
    photo_bytes = _pil_to_png_bytes(make_poster(payload=QR_BAD))

    res = audit_images(master_bytes, photo_bytes)
    assert res["qr_state"] == "MISMATCH DETECTED"
