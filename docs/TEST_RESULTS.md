# PosterProof — Test Results & Evaluation Summary

This document records the automated regression test outcomes, synthetic fixture evaluation results, web application smoke-check findings, and operational bounds for **PosterProof**.

---

## 1. Automated Unit Test Regression Results

All unit test cases were executed using `pytest 9.1.1` in Python 3.11:

```text
tests/test_core.py::test_decode_image_rejects_junk_bytes PASSED          [ 16%]
tests/test_core.py::test_crop_roi_simple_rectangle PASSED                [ 33%]
tests/test_core.py::test_decode_qr_reads_simple_qr PASSED                [ 50%]
tests/test_core.py::test_feature_register_refuses_low_feature_count PASSED [ 66%]
tests/test_core.py::test_audit_images_unchanged_scenario_pipeline PASSED [ 83%]
tests/test_core.py::test_audit_images_payload_mismatch PASSED            [100%]

============================== 6 passed in 3.60s ==============================
```

### Coverage Summary
- **Error Handling:** Verified `AuditError` raising on invalid/corrupt image bytes.
- **ROI Crop Logic:** Verified rectangle array slicing and bounds validation.
- **QR Decoding:** Verified OpenCV `QRCodeDetector` on cropped ROI arrays.
- **ORB Feature Registration:** Verified feature count threshold checks (<12 keypoints refusal).
- **Audit Pipeline:** Verified end-to-end audit execution for matched and tampered payloads.

---

## 2. Synthetic Fixture Scenario Outcomes

The table below summarizes observed evaluation results when running PosterProof against the 5 included synthetic poster fixtures:

| Demo Route | Reference Master | Candidate Photo Input | QR Status | Monitored Field Status | Verdict / Result |
|---|---|---|---|---|---|
| `/demo/scenario_1` | `approved_master.png` | `poster_changed_qr_photo.png` | `MISMATCH DETECTED` | `CHANGE CANDIDATE — REVIEW` | Tampered QR Payload Flagged |
| `/demo/scenario_2` | `approved_master.png` | `poster_changed_date_photo.png` | `QR PAYLOAD MATCHED` | `CHANGE CANDIDATE — REVIEW` | Date Change Flagged for Review |
| `/demo/scenario_3` | `approved_master.png` | `poster_unchanged_photo.png` | `QR PAYLOAD MATCHED` | `NO THRESHOLD-EXCEEDING CHANGE...` | Poster Integrity Verified |
| `/demo/scenario_4` | `approved_master.png` | `unrelated_poor_input.png` | N/A | N/A | Refusal: Insufficient Feature Matches |

*Note: Observations on this synthetic fixture set serve as prototype demonstration benchmarks and do not constitute a formal guarantee for uncalibrated real-world posters.*

---

## 3. Web Application Smoke-Check Summary

A local HTTP server smoke check was performed on `http://127.0.0.1:8765/`:

1. **Home Page (`/`):** Returns HTTP 200 containing the HTML upload form (`<form action="/audit"`).
2. **Demo Scenarios 1–3 (`/demo/scenario_1` .. `3`):** Return HTTP 200 with full audit report pages displaying master/photo ROI crops, difference metrics, and verification badges.
3. **Refusal Route (`/demo/scenario_4`):** Returns HTTP 400 with a user-friendly audit refusal card (`Audit Refusal / Verification Error`).

---

## 4. Evaluation Limits & Future Work

- **Scope:** PosterProof provides evidence visualization for human event-staff reviewers; it does not claim automated legal certification or general anti-tamper guarantee.
- **Camera & Capture Variability:** Real-world performance under severe glare, physical folds, extreme perspective tilt (>45°), or low lighting requires further empirical evaluation with a physical benchmark dataset.
- **Monitored Field Threshold Tuning:** Thresholds for changed pixel area and mean intensity differences should be calibrated to specific print media and camera sensors in production deployments.
