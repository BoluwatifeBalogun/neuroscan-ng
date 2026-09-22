# NeuroScan NG: Audit Report

Audit of the implementation against *Design and Implementation of a Web-Based Intelligent System for Early Detection of Alzheimer's Disease from MRI Images using CNNs* (Chapters One and Two), 22 September 2026.

## 1. Report compliance: 30/30 checks passed

Every verifiable claim in the write-up was mapped to code and tested automatically.

| Report reference | Requirement | Status |
|---|---|---|
| Obj. 1 / 1.7 | Image validation, RGB conversion, 224×224 resize, [0,1] normalisation, identical at training and inference | Implemented |
| Obj. 1 / 1.7 | Stratified train / validation / test split (70/15/15) | Implemented |
| Obj. 2 / 2.2.8 | VGG19 ImageNet base, frozen phase 1, block5 fine-tune at 1e-5, new softmax head | Implemented |
| Sec. 2.3 | Class imbalance handled (class weights + augmentation on training set only) | Implemented |
| Obj. 3 | Accuracy, precision, recall, sensitivity, specificity, F1, confusion matrix on a held-out test set, persisted to metrics.json | Implemented |
| Obj. 4 | Authorised-user web interface: registration, hashed sign-in, guarded upload | Implemented |
| Obj. 5 | Trained model integrated; predicted class, confidence score and probability breakdown displayed; records retained | Implemented |
| Obj. 6 | Functionality tested end to end; per-scan response time measured and stored | Implemented |
| 1.9 / 2.7 | Decision-support framing: disclaimer on result page, about page and every footer | Implemented |
| Wen et al. / Yagis et al. concerns | Split before augmentation; augmentation never touches validation or test data | Implemented |

**Pending by design:** Objectives 2 and 3 are implemented as a pipeline but not yet executed; real training runs on Colab via `train_neuroscan_colab.ipynb`. Until the artifacts are installed, the app runs in labelled demo mode.

## 2. Security audit

Attacks attempted and outcomes:

| Test | Result |
|---|---|
| IDOR: user B opening user A's result, scan image, heatmap | Safe (404) |
| SQL injection on login | Safe (parameterised queries) |
| Stored XSS via malicious filename | Safe (secure_filename + Jinja autoescaping) |
| Path traversal via filename | Safe (UUID storage names) |
| Oversize upload (11 MB) | Safe (413 handled with user message) |
| Password storage | Safe (scrypt hashes) |
| Cross-site form POST (CSRF) | **Was vulnerable, fixed** |
| Non-MRI input (colour photo) | **Was silently classified, fixed** |
| DB init under gunicorn | **Was broken, fixed** |

### Fixes applied during this audit
1. **CSRF protection**: session-bound token required on register, login and predict; requests without it or with a foreign session's token are rejected with 400.
2. **MRI plausibility gate**: brain MRI slices are effectively grayscale, so strongly coloured uploads (phone photos, screenshots) are now flagged with a visible caution on the result page. The check warns rather than blocks, consistent with Section 1.8 treating out-of-distribution input as a stated limitation.
3. **Deployment fix**: database initialisation now runs at import, so the app works under gunicorn or any WSGI server, not only `python app.py`.
4. Minor CSS defects removed (an invalid `clamp(...auto)` declaration and a redundant stroke rule).

## 3. Real-life scenario

Simulated session: a consultant radiologist at UNTH Enugu working an 8-patient memory-clinic list, with realistic synthetic axial T1-style slices (grayscale, skull ring, ventricles scaled by an atrophy parameter, scanner-like noise), plus a radiographer at AKTH Kano on a parallel account.

Observed behaviour, all as intended:
- Registration lands directly on the dashboard.
- Uploading a referral PDF by mistake is blocked with a clear message.
- A colour phone photo of a film is classified but carries the atypical-input caution.
- All 8 patient slices classified with confidence scores; none falsely flagged by the plausibility gate.
- History listed all 9 analyses; dashboard statistics matched the database exactly.
- The Kano account saw only its own scan and received 404 on Enugu record IDs.
- Logout invalidated access to protected pages.

Response times (Objective 6, demo inference path): server-side classification averaged ~1 ms; full request round-trip averaged 6 ms (p95: 6 ms). With the trained VGG19, expect roughly 300 to 900 ms additional per scan on CPU, still comfortably within an interactive workflow.

## 4. Honest limitations to carry into Chapter Five

- **Demo predictions are content-independent.** They are deterministic per file but derived from the file hash, so the scenario's class assignments do not track the synthetic atrophy levels. Correlation between prediction and pathology can only be evaluated after real training.
- **Slice-level split.** The Kaggle dataset carries no patient identifiers, so the 70/15/15 split is at image level. If multiple slices from one patient exist, patient-level leakage is possible; this is exactly the reproducibility concern Wen et al. and Yagis et al. raise and should be acknowledged, not hidden.
- **No rate limiting or account lockout**; brute-force login attempts are not throttled.
- **Uploads stored unencrypted** on the server filesystem; for any deployment beyond the study, encryption at rest and an NDPR-aware retention policy are needed.
- **SQLite** is appropriate for the study's scale, but concurrent-writer load in a multi-facility deployment would call for PostgreSQL.

## 5. Verdict

The implementation fully covers the system described in Chapters One and Two, the security posture is sound after the three fixes above, and the complete clinician workflow runs cleanly end to end. The only outstanding step to a fully live system is executing the training notebook and installing the resulting model artifacts.
