# PosterProof — Credits & Dependency Attributions

PosterProof relies on the following open-source software libraries and standard components:

## Core Dependencies

- **Python Standard Library** — `http.server`, `base64`, `html`, `os`, `io`
  - License: Python Software Foundation License (PSFL)
  - Purpose: HTTP web server foundation, multipart request parsing, Base64 encoding.

- **OpenCV (`opencv-python-headless`)** — OpenCV Team
  - License: Apache License 2.0
  - Purpose: Feature detection (ORB), descriptor matching (BFMatcher), perspective warping (homography RANSAC), and QR code decoding (`QRCodeDetector`).

- **NumPy (`numpy`)** — NumPy Developers
  - License: BSD 3-Clause Clear License
  - Purpose: In-memory image buffer arrays, matrix operations, difference metrics.

- **Pytest (`pytest`)** — Pytest Development Team
  - License: MIT License
  - Purpose: Automated test runner and assertion suite.

- **Pillow (`Pillow`)** — Jeffrey A. Clark & Contributors
  - License: HPND License
  - Purpose: Synthetic poster fixture generation and font drawing.

- **qrcode (`qrcode`)** — Lincoln Loop & Contributors
  - License: BSD License
  - Purpose: Synthetic QR code generation for demo fixtures and testing.

## License Review & Compliance

All included packages use permissive open-source licenses (MIT, Apache 2.0, BSD, PSFL, HPND) compatible with MIT distribution. No copyleft or restrictive licenses are included.

## AI assistance and model use

Google Antigravity IDE was used to assist with code drafting, test scaffolding, and technical documentation. The running prototype does not call a generative AI service and does not bundle a pretrained AI model. Disclose AI assistance in the format requested by the event rules.

## Licensing notes

The project code is released under the MIT License in `LICENSE`. Third-party packages retain their own licences; review the package metadata and licence texts for the exact versions installed in your environment before final submission. This file is a credit summary, not a substitute for third-party licence terms.