# Memora NG: implementation plan

Web-based system for early detection of Alzheimer's disease from MRI images using a
Vision Transformer (ViT) and a Long Short-Term Memory (LSTM) network. Case study: Nigeria.

## 1. Algorithm

Input: an ordered series of axial MRI slices for one patient (image files or one NIfTI volume).

1. **Validate** each slice: readable image, at least 64 px, greyscale, not blank, dark
   background around the head, plausible amount of tissue. Colour photos, documents
   and blank files are rejected. If more than 40% of a series fails, the upload is refused.
2. **Preprocess** each slice: greyscale, percentile intensity normalisation on tissue
   pixels, pad to a square, resize to 224 x 224.
3. **Resample** the series to T = 16 evenly spaced slices, keeping anatomical order.
   Fewer than 16 slices are repeated and the study is marked "limited context".
4. **Vision Transformer** (ViT-Tiny, patch 16, ImageNet pre-trained): each slice becomes
   196 patches plus a [CLS] token, passes through 12 self-attention blocks, and the
   [CLS] output (192 numbers) is the slice's feature vector. Output: 16 x 192.
5. **Bidirectional LSTM** (128 hidden units each way) reads the 16 vectors in order.
   Output: 16 x 256.
6. **Attention pooling**: a learned score per slice, softmax over the 16 slices, weighted sum.
7. **Classifier**: LayerNorm, dropout 0.3, linear layer, softmax over three classes:
   Cognitively Normal, Early Stage (MCI), Alzheimer's Disease.
   Confidence = the highest class probability.
8. **Explanation**: the 16 slice weights from step 6 and the ViT's last-layer [CLS]
   attention over the 14 x 14 patch grid are returned with the prediction.
9. **Store** the MRI_Images row, the Predictions row, the 16 prepared slices and
   attention overlays for the three highest-weighted slices.

Training (Colab, PyTorch): subject-level stratified 70/15/15 split, class-weighted
cross entropy (weights capped at 3), label smoothing 0.05, AdamW with one-cycle
schedule. Stage A: 4 epochs with the ViT frozen. Stage B: up to 14 epochs fine-tuning
the last 4 ViT blocks at a low learning rate, early stopping on validation macro F1.
Augmentation applies the same small rotation, scale and shift to every slice of a
series, plus brightness and gamma jitter, and draws a random ordered subset of slices.
Evaluation on the held-out test subjects: accuracy, precision, recall, F1-score,
ROC-AUC (one-vs-rest) and the confusion matrix. The model is exported to ONNX and
checked against PyTorch.

## 2. System flow (activity diagram, section 3.12.4)

    Open site -> Sign in -> authenticated?
        no  -> error, retry (5 failures locks the email for 10 minutes)
        yes -> Dashboard
    Select or register patient -> Upload MRI series -> valid?
        no  -> rejected with the reason, nothing stored
        yes -> Preprocess -> ViT features -> LSTM sequence -> Classify
            -> Save image record + prediction -> Show result and confidence
            -> View history / print report / add clinical note -> Sign out

## 3. Architecture (section 3.12.10)

| Layer | Implementation |
|---|---|
| Presentation | Jinja templates, Bootstrap grid, custom CSS and JavaScript |
| Application | Flask blueprints: public, auth, workspace, admin. `study_service.py` runs the pipeline |
| Artificial intelligence | `memora/ai`: preprocessing, ONNX Runtime engine, attention overlays. `training/`: PyTorch model and training |
| Database | MySQL through SQLAlchemy (SQLite fallback for local runs) |

Database tables (section 3.12.8): Users, Patients, MRI_Images, Predictions, plus Activity_Log.
One user manages many patients, one patient has many MRI series, one series has one prediction.

Actors and use cases (section 3.12.3): the Administrator registers users, manages
accounts and views the activity log. The Healthcare Professional signs in, manages
patients, uploads MRI, views results and history. There is no public sign-up.

## 4. Tools

Python 3.11, Flask, Flask-SQLAlchemy, PyMySQL, MySQL, ONNX Runtime, NumPy, Pillow,
nibabel, Gunicorn. Training: PyTorch, timm, scikit-learn. Front end: HTML, CSS,
JavaScript, Bootstrap grid. Git and GitHub. Testing: pytest.

## 5. Work plan (Agile iterations, section 3.13)

| Iteration | Output | Status |
|---|---|---|
| 1 Requirements | Read Chapters 1 to 3, list use cases, classes and tables | done |
| 2 AI core | Shared preprocessing, ViT-LSTM, training script, ONNX export, Colab notebook | done, pipeline verified end to end |
| 3 Application | Auth, roles, patients, upload, prediction, history, report, admin, logging | done |
| 4 Interface | Landing page, workspace, responsive layout, motion | done |
| 5 Testing | 26 automated tests, desktop and mobile visual checks | done |
| 6 Training | Run the notebook on real data, install the model, record metrics for Chapter 4 | to do (needs Colab GPU) |
| 7 Deployment | GitHub, Render or any host with MySQL | to do |

## 6. Front-end direction

Charcoal and gold. Plus Jakarta Sans with JetBrains Mono for figures, both self-hosted.
Design decisions were checked against the ui-ux-pro-max skill's rule set (contrast 4.5:1,
44 px touch targets, visible focus, reduced-motion support, labelled fields, errors
next to fields, bottom navigation with five items, no emoji icons). Interaction patterns
were written from scratch in the style of components catalogued on 21st.dev (spotlight
card, border beam, text shimmer, dock, marquee, number ticker, dot pattern) and of dark
Framer AI templates (floating pill navigation, split hero with a product visual, bento
grid, FAQ). No third-party component code is included.
