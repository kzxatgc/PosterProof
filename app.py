"""
PosterProof — Evidence-First Event Poster Audit.
Compare a printed poster photo against an approved master to surface evidence of changes.
"""

import base64
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os

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


# ---------------------------------------------------------------------------
# Audit Pipeline
# ---------------------------------------------------------------------------

DEFAULT_QR_ROI = (50, 400, 150, 150)
DEFAULT_FIELD_ROI = (50, 250, 400, 100)
DEFAULT_EXPECTED_PAYLOAD = "https://event.example.com/ticket"


def audit_images(
    master_bytes: bytes,
    photo_bytes: bytes,
    qr_roi: tuple = DEFAULT_QR_ROI,
    field_roi: tuple = DEFAULT_FIELD_ROI,
    expected_qr: str = DEFAULT_EXPECTED_PAYLOAD,
) -> dict:
    """Perform full audit pipeline: decode -> register -> crop ROIs -> decode QR -> compare field -> assemble result."""
    master_img = decode_image(master_bytes, "Master image")
    photo_img = decode_image(photo_bytes, "Photo image")

    # Align photo to master geometry
    registered_img = register_image(master_img, photo_img)

    # 1. QR code audit (ROI crop ONLY)
    master_qr_crop = crop_roi(master_img, qr_roi, "Master QR ROI")
    photo_qr_crop = crop_roi(registered_img, qr_roi, "Photo QR ROI")

    master_qr_text = decode_qr(master_qr_crop)
    photo_qr_text = decode_qr(photo_qr_crop)

    if not photo_qr_text:
        qr_state = "COULD NOT VERIFY"
        qr_match = False
    else:
        target_payload = expected_qr if expected_qr else master_qr_text
        if photo_qr_text == target_payload:
            qr_state = "QR PAYLOAD MATCHED"
            qr_match = True
        else:
            qr_state = "MISMATCH DETECTED"
            qr_match = False

    # 2. Monitored Field difference audit
    ref_field_crop = crop_roi(master_img, field_roi, "Master Field ROI")
    obs_field_crop = crop_roi(registered_img, field_roi, "Photo Field ROI")

    ref_gray = cv2.GaussianBlur(cv2.cvtColor(ref_field_crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    obs_gray = cv2.GaussianBlur(cv2.cvtColor(obs_field_crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    diff = cv2.absdiff(ref_gray, obs_gray)
    changed_pct = round(100.0 * float(np.mean(diff > 30)), 2)
    mean_pct = round(100.0 * float(np.mean(diff)) / 255.0, 2)

    is_field_changed = changed_pct >= 7.5 or mean_pct >= 4.0
    field_state = (
        "CHANGE CANDIDATE — REVIEW"
        if is_field_changed
        else "NO THRESHOLD-EXCEEDING CHANGE FLAGGED IN MONITORED REGION"
    )

    diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

    return {
        "qr_state": qr_state,
        "qr_match": qr_match,
        "master_qr_payload": master_qr_text,
        "photo_qr_payload": photo_qr_text,
        "expected_qr_payload": expected_qr,
        "field_state": field_state,
        "field_changed": is_field_changed,
        "changed_pct": changed_pct,
        "mean_pct": mean_pct,
        "master_uri": _to_data_uri(master_img),
        "registered_photo_uri": _to_data_uri(registered_img),
        "master_qr_crop_uri": _to_data_uri(master_qr_crop),
        "photo_qr_crop_uri": _to_data_uri(photo_qr_crop),
        "master_field_crop_uri": _to_data_uri(ref_field_crop),
        "photo_field_crop_uri": _to_data_uri(obs_field_crop),
        "diff_crop_uri": _to_data_uri(diff_colored),
    }


# ---------------------------------------------------------------------------
# HTML Presentation Shell — Neo-Pop Digital Aesthetic
# ---------------------------------------------------------------------------

CSS_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;900&family=Space+Grotesk:wght@500;700&display=swap');

:root {
  --bg-base: #F8F9FA;
  --text-main: #0A0A0A;
  --primary-pink: #FF2A85;
  --primary-cyan: #00F0FF;
  --primary-purple: #B829FF;
  --neon-yellow: #FFE600;
  --neon-green: #00FF66;
  --card-bg: #FFFFFF;
  --border-dark: #0A0A0A;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Space Grotesk', sans-serif;
  background-color: var(--bg-base);
  color: var(--text-main);
  line-height: 1.5;
  padding: 2.5rem 1rem;
  background-image: radial-gradient(#D1D5DB 1.5px, transparent 1.5px);
  background-size: 24px 24px;
}

.container { max-width: 1000px; margin: 0 auto; }

header {
  text-align: center;
  margin-bottom: 2.5rem;
  position: relative;
}

h1 {
  font-family: 'Plus Jakarta Sans', sans-serif;
  font-size: 3.2rem;
  font-weight: 900;
  color: #FFFFFF;
  text-transform: uppercase;
  letter-spacing: -1px;
  -webkit-text-stroke: 3px #0A0A0A;
  text-shadow: 5px 5px 0px #0A0A0A;
  margin-bottom: 0.5rem;
}

p.subtitle {
  font-size: 1.1rem;
  font-weight: 700;
  background: var(--neon-yellow);
  display: inline-block;
  padding: 0.25rem 1rem;
  border: 3px solid var(--border-dark);
  border-radius: 999px;
  box-shadow: 3px 3px 0px var(--border-dark);
  transform: rotate(-1deg);
}

.card {
  background: var(--card-bg);
  border: 3px solid var(--border-dark);
  border-radius: 20px;
  padding: 2rem;
  margin-bottom: 2rem;
  box-shadow: 6px 6px 0px var(--border-dark);
  position: relative;
}

.card h2 {
  font-family: 'Plus Jakarta Sans', sans-serif;
  font-size: 1.4rem;
  font-weight: 900;
  color: var(--text-main);
  margin-bottom: 1.25rem;
  border-bottom: 3px solid var(--border-dark);
  padding-bottom: 0.75rem;
  text-transform: uppercase;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.card h2::before {
  content: "✨";
}

.form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
@media (max-width: 768px) { .form-grid { grid-template-columns: 1fr; } }

.form-group { margin-bottom: 1.25rem; }

label {
  display: block;
  font-weight: 700;
  margin-bottom: 0.5rem;
  color: var(--text-main);
  font-size: 0.95rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

input[type="file"], input[type="text"], input[type="number"] {
  width: 100%;
  padding: 0.75rem 1rem;
  border-radius: 10px;
  border: 3px solid var(--border-dark);
  background: #FFF;
  color: var(--text-main);
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 1rem;
  box-shadow: 3px 3px 0px var(--border-dark);
  transition: transform 0.1s, box-shadow 0.1s;
}

input[type="file"]:focus, input[type="text"]:focus, input[type="number"]:focus {
  outline: none;
  transform: translate(-2px, -2px);
  box-shadow: 5px 5px 0px var(--border-dark);
  border-color: var(--primary-pink);
}

.roi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; }

.btn {
  display: inline-block;
  background: var(--primary-pink);
  color: #FFF;
  border: 3px solid var(--border-dark);
  padding: 0.9rem 2rem;
  font-family: 'Plus Jakarta Sans', sans-serif;
  font-weight: 900;
  font-size: 1.1rem;
  text-transform: uppercase;
  border-radius: 12px;
  cursor: pointer;
  text-decoration: none;
  text-align: center;
  box-shadow: 4px 4px 0px var(--border-dark);
  transition: all 0.1s ease;
}

.btn:hover {
  transform: translate(-3px, -3px);
  box-shadow: 7px 7px 0px var(--border-dark);
  background: #ff0072;
}

.demo-links { display: flex; flex-wrap: wrap; gap: 0.75rem; margin-top: 1rem; }

.demo-btn {
  background: var(--primary-cyan);
  color: var(--border-dark);
  border: 3px solid var(--border-dark);
  padding: 0.6rem 1.2rem;
  border-radius: 10px;
  text-decoration: none;
  font-weight: 700;
  font-size: 0.9rem;
  box-shadow: 3px 3px 0px var(--border-dark);
  transition: all 0.1s ease;
}

.demo-btn:hover {
  transform: translate(-2px, -2px);
  box-shadow: 5px 5px 0px var(--border-dark);
  background: #00d2e0;
}

.badge {
  display: inline-block;
  padding: 0.4rem 0.9rem;
  border-radius: 8px;
  font-weight: 900;
  font-size: 0.85rem;
  text-transform: uppercase;
  border: 2px solid var(--border-dark);
  box-shadow: 2px 2px 0px var(--border-dark);
}

.badge-success { background: var(--neon-green); color: var(--border-dark); }
.badge-warning { background: var(--neon-yellow); color: var(--border-dark); }
.badge-danger { background: var(--primary-pink); color: #FFF; }
.badge-info { background: var(--primary-cyan); color: var(--border-dark); }

.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }
.grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.25rem; }
@media (max-width: 768px) { .grid-2, .grid-3 { grid-template-columns: 1fr; } }

.img-box {
  text-align: center;
  background: #FFF;
  padding: 1rem;
  border-radius: 14px;
  border: 3px solid var(--border-dark);
  box-shadow: 4px 4px 0px var(--border-dark);
}

.img-box img {
  max-width: 100%;
  height: auto;
  border-radius: 8px;
  border: 2px solid var(--border-dark);
  display: block;
  margin: 0 auto 0.75rem;
}

.img-box span {
  font-size: 0.85rem;
  color: var(--text-main);
  font-weight: 700;
  text-transform: uppercase;
}

table.metrics { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
table.metrics th, table.metrics td {
  padding: 0.8rem;
  border-bottom: 2px solid var(--border-dark);
  text-align: left;
  font-size: 0.95rem;
}
table.metrics th { color: var(--text-main); font-weight: 900; text-transform: uppercase; background: #F3F4F6; }
table.metrics td code {
  background: var(--primary-cyan);
  padding: 0.2rem 0.4rem;
  border-radius: 4px;
  border: 1.5px solid var(--border-dark);
  font-weight: 700;
}
</style>
"""


def input_form() -> str:
    """Return the HTML upload form and demo scenarios."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PosterProof — Event Poster Audit</title>
  {CSS_STYLES}
</head>
<body>
  <div class="container">
    <header>
      <h1>PosterProof</h1>
      <p class="subtitle">Evidence-First Event Poster Audit & Integrity Verification</p>
    </header>

    <div class="card">
      <h2>Upload Poster Images for Audit</h2>
      <form action="/audit" method="post" enctype="multipart/form-data">
        <div class="form-grid">
          <div class="form-group">
            <label for="master">Approved Master Poster (PNG/JPG):</label>
            <input type="file" id="master" name="master" accept="image/*" required>
          </div>
          <div class="form-group">
            <label for="photo">Printed Poster Photo (PNG/JPG):</label>
            <input type="file" id="photo" name="photo" accept="image/*" required>
          </div>
        </div>

        <div class="form-group" style="margin-top: 1rem;">
          <label for="expected_qr">Expected QR Payload (URL or String):</label>
          <input type="text" id="expected_qr" name="expected_qr" value="{DEFAULT_EXPECTED_PAYLOAD}">
        </div>

        <div class="form-grid" style="margin-top: 1rem;">
          <div class="form-group">
            <label>Monitored QR Code Region (X, Y, Width, Height):</label>
            <div class="roi-grid">
              <input type="number" name="qr_x" value="{DEFAULT_QR_ROI[0]}" required>
              <input type="number" name="qr_y" value="{DEFAULT_QR_ROI[1]}" required>
              <input type="number" name="qr_w" value="{DEFAULT_QR_ROI[2]}" required>
              <input type="number" name="qr_h" value="{DEFAULT_QR_ROI[3]}" required>
            </div>
          </div>

          <div class="form-group">
            <label>Monitored Text/Date Region (X, Y, Width, Height):</label>
            <div class="roi-grid">
              <input type="number" name="field_x" value="{DEFAULT_FIELD_ROI[0]}" required>
              <input type="number" name="field_y" value="{DEFAULT_FIELD_ROI[1]}" required>
              <input type="number" name="field_w" value="{DEFAULT_FIELD_ROI[2]}" required>
              <input type="number" name="field_h" value="{DEFAULT_FIELD_ROI[3]}" required>
            </div>
          </div>
        </div>

        <div style="margin-top: 1.75rem; text-align: center;">
          <button type="submit" class="btn">Run Poster Proof Audit</button>
        </div>
      </form>
    </div>

    <div class="card">
      <h2>Interactive Demo Scenarios</h2>
      <p style="font-size: 0.95rem; font-weight: 700; margin-bottom: 1rem;">
        Click any scenario below to test PosterProof on synthetic poster fixtures:
      </p>
      <div class="demo-links">
        <a href="/demo/scenario_1" class="demo-btn">Scenario 1: QR Code Tampered</a>
        <a href="/demo/scenario_2" class="demo-btn">Scenario 2: Date/Text Changed</a>
        <a href="/demo/scenario_3" class="demo-btn">Scenario 3: Unchanged Photo (Match)</a>
        <a href="/demo/scenario_4" class="demo-btn">Scenario 4: Poor Input / Unrelated Poster</a>
      </div>
    </div>
  </div>
</body>
</html>
"""


def result_html(res: dict) -> str:
    """Return the audit evidence result page HTML."""
    if res["qr_state"] == "QR PAYLOAD MATCHED":
        qr_badge = "badge-success"
    elif res["qr_state"] == "MISMATCH DETECTED":
        qr_badge = "badge-danger"
    else:
        qr_badge = "badge-warning"

    if "CHANGE CANDIDATE" in res["field_state"]:
        field_badge = "badge-danger"
    else:
        field_badge = "badge-success"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Audit Report — PosterProof</title>
  {CSS_STYLES}
</head>
<body>
  <div class="container">
    <header>
      <h1>Audit Report</h1>
      <p class="subtitle">Evidence Summary & Visual Alignment Analysis</p>
    </header>

    <div class="card">
      <h2>Audit Findings Summary</h2>
      <table class="metrics">
        <tr>
          <th>Check Item</th>
          <th>Status / Measured Value</th>
        </tr>
        <tr>
          <td><strong>QR Code Verification</strong></td>
          <td><span class="badge {qr_badge}">{html.escape(res['qr_state'])}</span></td>
        </tr>
        <tr>
          <td>Master QR Payload</td>
          <td><code>{html.escape(res.get('master_qr_payload', '') or 'None')}</code></td>
        </tr>
        <tr>
          <td>Photo QR Payload</td>
          <td><code>{html.escape(res.get('photo_qr_payload', '') or 'None')}</code></td>
        </tr>
        <tr>
          <td>Expected Payload</td>
          <td><code>{html.escape(res.get('expected_qr_payload', '') or 'None')}</code></td>
        </tr>
        <tr>
          <td><strong>Monitored Text Region</strong></td>
          <td><span class="badge {field_badge}">{html.escape(res['field_state'])}</span></td>
        </tr>
        <tr>
          <td>Changed Pixel Area</td>
          <td><strong>{res['changed_pct']}%</strong> (Threshold: &ge; 7.5%)</td>
        </tr>
        <tr>
          <td>Mean Intensity Difference</td>
          <td><strong>{res['mean_pct']}%</strong> (Threshold: &ge; 4.0%)</td>
        </tr>
      </table>
    </div>

    <div class="card">
      <h2>Visual Alignment & Rectification</h2>
      <div class="grid-2">
        <div class="img-box">
          <img src="{res['master_uri']}" alt="Master Poster">
          <span>Approved Master Reference</span>
        </div>
        <div class="img-box">
          <img src="{res['registered_photo_uri']}" alt="Rectified Photo">
          <span>Rectified Photo</span>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>Monitored Region Evidence Crops</h2>
      <div class="grid-3" style="margin-bottom: 1.5rem;">
        <div class="img-box">
          <img src="{res['master_qr_crop_uri']}" alt="Master QR Crop">
          <span>Master QR ROI</span>
        </div>
        <div class="img-box">
          <img src="{res['photo_qr_crop_uri']}" alt="Photo QR Crop">
          <span>Photo QR ROI</span>
        </div>
        <div class="img-box" style="display: flex; align-items: center; justify-content: center; flex-direction: column;">
          <span><strong>QR Match Status</strong></span>
          <span class="badge {qr_badge}" style="margin-top: 0.5rem;">{html.escape(res['qr_state'])}</span>
        </div>
      </div>

      <div class="grid-3">
        <div class="img-box">
          <img src="{res['master_field_crop_uri']}" alt="Master Field Crop">
          <span>Master Text ROI</span>
        </div>
        <div class="img-box">
          <img src="{res['photo_field_crop_uri']}" alt="Photo Field Crop">
          <span>Photo Text ROI</span>
        </div>
        <div class="img-box">
          <img src="{res['diff_crop_uri']}" alt="Difference Heatmap">
          <span>Difference Heatmap</span>
        </div>
      </div>
    </div>

    <div style="text-align: center; margin-top: 1.75rem;">
      <a href="/" class="btn">&larr; Run Another Audit</a>
    </div>
  </div>
</body>
</html>
"""


def error_html(message: str) -> str:
    """Return styled error page for AuditError and HTTP error states."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Audit Error — PosterProof</title>
  {CSS_STYLES}
</head>
<body>
  <div class="container">
    <header>
      <h1>PosterProof</h1>
      <p class="subtitle">Evidence-First Event Poster Audit</p>
    </header>
    <div class="card" style="border-color: var(--primary-pink);">
      <h2 style="color: var(--primary-pink);">Audit Refusal / Verification Error</h2>
      <p style="margin-bottom: 1rem; color: var(--text-main); font-size: 1.05rem; font-weight: 700;">
        {html.escape(message)}
      </p>
      <div style="text-align: center; margin-top: 1.75rem;">
        <a href="/" class="btn">&larr; Return to Upload Form</a>
      </div>
    </div>
  </div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP Multipart Parser & Request Handler
# ---------------------------------------------------------------------------

def parse_multipart(body_bytes: bytes, boundary: bytes) -> tuple:
    """Parse raw multipart/form-data body bytes into fields (dict[str, str]) and files (dict[str, bytes])."""
    fields = {}
    files = {}

    delimiter = b"--" + boundary
    parts = body_bytes.split(delimiter)

    for part in parts:
        if not part or part == b"--\r\n" or part == b"--" or part.startswith(b"--"):
            continue
        if b"\r\n\r\n" not in part:
            continue
        header_section, content = part.split(b"\r\n\r\n", 1)
        if content.endswith(b"\r\n"):
            content = content[:-2]

        header_text = header_section.decode("utf-8", errors="ignore")
        headers = {}
        for line in header_text.split("\r\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()

        cd = headers.get("content-disposition", "")
        if "name=" in cd:
            name_part = cd.split('name="')[1].split('"')[0]
            if 'filename="' in cd:
                files[name_part] = content
            else:
                fields[name_part] = content.decode("utf-8", errors="ignore")

    return fields, files


class Handler(BaseHTTPRequestHandler):
    """HTTP Handler supporting web upload form, /audit POST endpoint, and demo routes."""

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(input_form().encode("utf-8"))
        elif self.path.startswith("/demo/scenario_"):
            self.handle_demo_route(self.path)
        else:
            self.send_error(404, "Page Not Found")

    def do_POST(self):
        if self.path != "/audit":
            self.send_error(404, "Not Found")
            return

        content_type = self.headers.get("Content-Type", "")
        if "boundary=" not in content_type:
            self.send_response(400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(error_html("Invalid form submission.").encode("utf-8"))
            return

        boundary = content_type.split("boundary=")[1].split(";")[0].strip().encode("ascii")

        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            status_code = 413 if length > MAX_UPLOAD_BYTES else 400
            self.send_response(status_code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            err_msg = "File exceeds the 20 MB upload size limit." if status_code == 413 else "Invalid request body length."
            self.wfile.write(error_html(err_msg).encode("utf-8"))
            return

        body = self.rfile.read(length)
        fields, files = parse_multipart(body, boundary)

        master_bytes = files.get("master", b"")
        photo_bytes = files.get("photo", b"")

        if not master_bytes or not photo_bytes:
            self.send_response(400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(error_html("Both approved master poster and printed photo are required.").encode("utf-8"))
            return

        try:
            qr_roi = _roi_from_fields(fields, "qr") if "qr_x" in fields else DEFAULT_QR_ROI
            field_roi = _roi_from_fields(fields, "field") if "field_x" in fields else DEFAULT_FIELD_ROI
            expected_qr = fields.get("expected_qr", DEFAULT_EXPECTED_PAYLOAD)

            result = audit_images(master_bytes, photo_bytes, qr_roi, field_roi, expected_qr)
            html_out = result_html(result)

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_out.encode("utf-8"))
        except AuditError as exc:
            self.send_response(400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(error_html(str(exc)).encode("utf-8"))

    def handle_demo_route(self, path: str):
        """Serve one of 4 demo scenarios by reading pre-generated fixture PNGs."""
        _SCENARIO_MAP = {
            "/demo/scenario_1": (
                "approved_master.png",
                "scenario_3_qr_changed_photo.png",
                DEFAULT_EXPECTED_PAYLOAD,
            ),
            "/demo/scenario_2": (
                "approved_master.png",
                "scenario_2_date_changed_photo.png",
                DEFAULT_EXPECTED_PAYLOAD,
            ),
            "/demo/scenario_3": (
                "approved_master.png",
                "scenario_1_unchanged_photo.png",
                DEFAULT_EXPECTED_PAYLOAD,
            ),
            "/demo/scenario_4": (
                "approved_master.png",
                "poor_input_edge_case.png",
                DEFAULT_EXPECTED_PAYLOAD,
            ),
        }

        scenario = _SCENARIO_MAP.get(path)
        if scenario is None:
            self.send_error(404, "Demo route not found.")
            return

        master_filename, photo_filename, expected_qr = scenario
        samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples")
        master_path = os.path.join(samples_dir, master_filename)
        photo_path = os.path.join(samples_dir, photo_filename)

        missing = [f for f in (master_path, photo_path) if not os.path.isfile(f)]
        if missing:
            missing_names = ", ".join(os.path.basename(f) for f in missing)
            body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Demo Fixtures Not Ready — PosterProof</title>
  {CSS_STYLES}
</head>
<body>
  <div class="container">
    <header>
      <h1>PosterProof</h1>
      <p class="subtitle">Evidence-First Event Poster Audit</p>
    </header>
    <div class="card" style="border-color: var(--neon-yellow);">
      <h2 style="color: var(--text-main);">Demo Fixtures Not Generated Yet</h2>
      <p style="margin-bottom: 1rem; color: var(--text-main); font-weight: 700;">
        The following fixture file(s) have not been generated yet:
        <code>{html.escape(missing_names)}</code>
      </p>
      <p style="color: var(--text-main); font-size: 0.95rem; font-weight: 700;">
        Run <code>python tools/generate_samples.py</code> from the project root to create all demo fixtures.
      </p>
      <div style="text-align: center; margin-top: 1.75rem;">
        <a href="/" class="btn">&larr; Return to Upload Form</a>
      </div>
    </div>
  </div>
</body>
</html>"""
            self.send_response(404)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            return

        try:
            with open(master_path, "rb") as f:
                master_bytes = f.read()
            with open(photo_path, "rb") as f:
                photo_bytes = f.read()

            result = audit_images(master_bytes, photo_bytes, expected_qr=expected_qr)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(result_html(result).encode("utf-8"))
        except AuditError as exc:
            self.send_response(400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(error_html(str(exc)).encode("utf-8"))


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

HOST = "127.0.0.1"
PORT = int(os.environ.get("POSTERPROOF_PORT", "8765"))


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print("=" * 55)
    print("  PosterProof — Evidence-First Event Poster Audit")
    print("=" * 55)
    print(f"  Running at:  http://{HOST}:{PORT}/")
    print("  Press Ctrl+C to stop.")
    print("=" * 55)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")