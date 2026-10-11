# PosterProof — Evidence-First Event Poster Audit

**PosterProof** is a local, evidence-first web application designed for event staff to verify the integrity of printed event posters before public deployment. By comparing a photo of a printed poster against an approved master artwork, PosterProof automatically registers visual geometry, verifies QR code payloads, and flags unapproved changes in monitored text areas (such as event dates and times).

---

## Key Features

1. **In-Memory Local Processing:** All image uploads and comparison calculations remain strictly in memory; no uploaded images are written to temporary disk files.
2. **ORB Feature Registration & Homography Rectification:** Automatically aligns perspective-distorted poster photos taken at an angle back to reference master coordinates using OpenCV ORB feature detection and RANSAC homography.
3. **Cropped ROI QR Audit:** Scans and decodes QR codes exclusively from cropped region-of-interest (ROI) rectangles—never running unconstrained whole-image searches.
4. **Monitored Field Difference Analysis:** Evaluates monitored text fields using blurred grayscale difference metrics (reporting changed pixel percentages and mean intensity differences).
5. **Interactive Neo-Pop UI:** Clean, responsive web application served directly via Python's standard-library HTTP server.
6. **Pre-Built Interactive Demo Scenarios:** Includes 4 built-in demo scenarios exercising unchanged posters, tampered QR codes, altered dates, and low-feature refusal cases.
7. **Conservative Verification Refusal:** Safely refuses verification when visual features are insufficient (<12 matches) or registration evidence is weak, avoiding false confidence.
8. **Zero Mandatory External Server Dependencies:** Fully runnable offline on standard Python 3.10+ environments.

---

## Computer Vision Approach

PosterProof uses traditional computer-vision and image-processing techniques to compare a photographed event poster with its approved master artwork. The verification pipeline combines image registration, QR payload comparison, and monitored-region difference analysis.

### 1. Feature Detection and Perspective Alignment

A photograph taken at an angle may have a different perspective from the original poster. PosterProof uses OpenCV's ORB (Oriented FAST and Rotated BRIEF) feature detector to identify distinctive visual keypoints and descriptors in the images.

The descriptors are matched between the reference and photographed poster. Homography estimation with RANSAC is then used to estimate a perspective transformation and warp the photograph into the reference image's coordinate system.

When there are too few usable features or the registration evidence is unreliable, the application refuses the comparison rather than treating an uncertain alignment as trustworthy.

### 2. Region-Based QR Payload Verification

After establishing the relevant image coordinates, PosterProof examines the configured QR-code region of interest (ROI). OpenCV's QR-code detector attempts to decode the code within that region, and the resulting payload is compared with the registered reference payload.

The application deliberately restricts the audit to the configured region instead of searching the entire poster for another QR code. This keeps the check focused on the expected location.

A different decoded payload is reported as a QR mismatch. A matching payload establishes only that the decoded content matches the reference; it does not prove that the entire poster is unchanged or that the destination is safe.

### 3. Monitored-Region Difference Analysis

PosterProof also examines selected visual regions, such as an event date or time, independently of the QR audit.

The comparison uses blurred grayscale images and image-difference metrics, including the percentage of pixels exceeding the configured difference threshold and the mean intensity difference.

These measurements help identify regions that may have changed. A flagged region is a **change candidate for human review**, not automatic proof of tampering. Lighting changes, image quality, alignment errors, and other capture conditions can affect the measurements.

### 4. Evidence-Based Outcomes

The pipeline keeps its findings separate so that users can interpret them more carefully:

- **QR mismatch:** The decoded QR payload differs from the reference.
- **Visual change candidate:** A monitored region exceeds the configured difference threshold and needs human review.
- **Unchanged sample:** The tested alignment, QR payload, and monitored-region checks satisfy the application's configured criteria.
- **Verification refusal:** The available image features or registration evidence are insufficient for a reliable comparison.

These outcomes describe the checks that were performed; they do not provide a universal guarantee that a poster is authentic or unchanged.

### 5. Computer Vision, Not a Pretrained AI Model

The current prototype's core analysis uses established computer-vision and image-processing algorithms implemented with OpenCV and NumPy. It does not run a generative AI service or include a pretrained machine-learning model.

This makes the pipeline suitable for a self-contained local demonstration, but it also defines the current scope: PosterProof relies on feature matching, configured regions, difference metrics, and thresholds rather than a learned model trained to recognise every form of poster tampering.

### 6. Current Limitations

The reliability of the comparison depends on factors such as image quality, perspective, lighting, distinctive visual features, and the accuracy of the configured regions. An altered poster may also contain changes outside the monitored regions.

The current synthetic demonstration scenarios and automated tests exercise the intended pipeline behaviour; they do not constitute real-world security validation. Further evaluation using diverse real-world photographs would be required before making broader claims about detection accuracy.

PosterProof is designed as a human-review aid. It does not independently authenticate the master artwork, prove that every part of the photographed poster is unchanged, or determine whether a QR-code destination is safe.

---

## Quick Start Guide

### Prerequisites
- **Python 3.10 or newer** (`python --version` or `py -3 --version`)
- **Git**

### Installation & Setup

#### On Windows (PowerShell):
```powershell
# Clone the repository
git clone https://github.com/kzxatgc/PosterProof.git
cd PosterProof

# Create and activate virtual environment
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

If PowerShell execution policy blocks activation script execution:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
```

#### On macOS / Linux / WSL:
```bash
# Clone the repository
git clone https://github.com/kzxatgc/PosterProof.git
cd PosterProof

# Create and activate virtual environment
python3 -m venv .venv-unix
source .venv-unix/bin/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

---

## Running Automated Tests

Run the full pytest suite to verify core pipeline functionality:

```bash
python -m pytest -v
```

All unit test cases should pass cleanly (`6 passed`).

---

## Launching the Application

To start the local PosterProof web server:

```bash
python app.py
```

Output:
```text
PosterProof running at http://127.0.0.1:8765/
```

1. Open your browser and navigate to **`http://127.0.0.1:8765/`**.
2. Select files for Master Poster and Poster Photo, or click any of the **Interactive Demo Scenarios** to test pre-built synthetic poster fixtures.
3. Press **Ctrl+C** in the terminal to stop the server when finished.

---

## Repository Structure

```text
PosterProof/
├── app.py                      # Main HTTP server, audit pipeline, and HTML templates
├── requirements.txt            # Production runtime dependencies (numpy, opencv-python-headless)
├── requirements-dev.txt        # Development dependencies (pytest, Pillow, qrcode)
├── LICENSE                     # MIT License
├── README.md                   # Application documentation and quick start guide
├── run_windows.bat             # Windows runner script
├── run_unix.sh                 # Unix/Linux/macOS launcher script
├── docs/
│   └── CREDITS.md              # Dependency attributions and license review
├── data/
│   └── samples/                # Pre-generated synthetic demo PNG poster fixtures
├── tests/
│   └── test_core.py            # Unit test suite
└── tools/
    └── generate_samples.py     # Deterministic sample poster generator
```

---

## Author & Project Info

- **Author:** kzxatgc (Solo Developer)
- **Authorship:** kzxatgc


---

## Links & Submissions

- **Public Repository:** https://github.com/kzxatgc/PosterProof.git


---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
