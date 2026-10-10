"""Core tests for PosterProof application."""

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


def _generate_synthetic_poster(qr_text: str = DEFAULT_EXPECTED_PAYLOAD, date_text: str = "OCTOBER 15, 2026") -> bytes:
    """Helper to draw a 600x600 synthetic poster with rich features, text, and QR ROI."""
    img = np.full((600, 600, 3), 245, dtype=np.uint8)

    # Draw high-contrast shapes/grid so ORB feature detection finds > 12 keypoints easily
    np.random.seed(12345)
    for _ in range(40):
        pt1 = (int(np.random.randint(0, 600)), int(np.random.randint(0, 600)))
        pt2 = (int(np.random.randint(0, 600)), int(np.random.randint(0, 600)))
        color = (int(np.random.randint(0, 200)), int(np.random.randint(0, 200)), int(np.random.randint(0, 200)))
        cv2.line(img, pt1, pt2, color, 2)
        cv2.circle(img, pt1, 8, color, -1)

    # Draw date text inside field ROI (50, 250, 400, 100) -> x: 50..450, y: 250..350
    cv2.putText(img, f"DATE: {date_text}", (70, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)

    # Generate QR code for QR ROI (50, 400, 150, 150) -> x: 50..200, y: 400..550
    qr = qrcode.QRCode(box_size=3, border=2)
    qr.add_data(qr_text)
    qr.make(fit=True)
    qr_pil = qr.make_image(fill_color="black", back_color="white")
    qr_bgr = cv2.cvtColor(np.array(qr_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    qr_resized = cv2.resize(qr_bgr, (150, 150))
    img[400:550, 50:200] = qr_resized

    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_audit_images_unchanged_scenario_pipeline():
    """Verify audit_images reports matched QR and unchanged field on identical posters."""
    poster_bytes = _generate_synthetic_poster()
    res = audit_images(poster_bytes, poster_bytes)

    assert res["qr_state"] == "QR PAYLOAD MATCHED"
    assert res["field_state"] == "NO THRESHOLD-EXCEEDING CHANGE FLAGGED IN MONITORED REGION"


def test_audit_images_payload_mismatch():
    """Verify audit_images reports MISMATCH DETECTED when QR payloads differ."""
    master_bytes = _generate_synthetic_poster(qr_text="https://event.example.com/ticket")
    photo_bytes = _generate_synthetic_poster(qr_text="https://malicious-scam-site.com")

    res = audit_images(master_bytes, photo_bytes)
    assert res["qr_state"] == "MISMATCH DETECTED"
