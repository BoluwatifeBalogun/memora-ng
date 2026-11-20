# Memora NG

Web-based system for early detection of Alzheimer's disease from MRI images using a
hybrid Vision Transformer and LSTM network. Flask, MySQL, PyTorch, ONNX Runtime.

Read `PLAN.md` for the algorithm and design, and `WRITEUP_NOTES.md` for corrections
to the write-up.

## Run it locally

    pip install -r requirements.txt
    python run.py                      # http://localhost:5000

First-run logins (development only, change them before deployment):

| Role | Email | Password |
|---|---|---|
| Administrator | admin@memora.ng | Memora@2026 |
| Healthcare professional | clinician@memora.ng | Clinic@2026 |

Optional sample data (invented patients, synthetic slices):

    python scripts/seed_demo.py        # add
    python scripts/seed_demo.py --clear

`samples/` holds two synthetic slice series you can upload to try the Analyse page.
They are drawn shapes, not MRI data.

## Important: the model is not trained yet

Until a trained model is installed the app runs a **demonstration rule** and says so
on every screen, record and report. It has no accuracy and must not be presented as
the ViT-LSTM's result.

To train:

1. Zip this folder as `memora-ng.zip`.
2. Open `training/memora_train_colab.ipynb` in Google Colab, choose a T4 GPU, run all cells.
3. Put the downloaded `memora_vit_lstm.onnx` and `model_card.json` in `model_store/`.
4. Restart. The badge turns green and the Model page shows accuracy, precision,
   recall, F1-score, ROC-AUC and the confusion matrix from the test set.

The training code was run end to end here on synthetic data to confirm that training,
evaluation and ONNX export work and that ONNX Runtime matches PyTorch. It has not been
run on the real dataset, and the Kaggle folder layout could not be checked from the
build environment, so watch the "series found" count in the notebook's data cell.

## MySQL

    CREATE DATABASE memora CHARACTER SET utf8mb4;
    export DATABASE_URL="mysql+pymysql://user:password@localhost:3306/memora"
    python run.py                      # tables are created on first start

Without `DATABASE_URL` the app uses a SQLite file in `instance/`.

## Deploy

`render.yaml` and `Procfile` are included. Set `MEMORA_ENV=production`, `SECRET_KEY`,
`ADMIN_EMAIL`, `ADMIN_PASSWORD` and `DATABASE_URL` (see `.env.example`). On hosts with
temporary disks, uploaded slices are lost on redeploy unless a persistent disk is
attached and `UPLOAD_DIR` points to it.

## Tests

    pip install pytest
    python -m pytest tests -q          # 26 tests

## Layout

    memora/            Flask application
      ai/              preprocessing, inference engines, attention overlays
      blueprints/      public, auth, workspace, admin routes
      templates/ static/
    training/          ViT-LSTM model, dataset loader, training, ONNX export, Colab notebook
    model_store/       trained model files go here
    scripts/           synthetic phantoms and sample data
    tests/

Fonts (Plus Jakarta Sans, JetBrains Mono) are bundled under the SIL Open Font License.
Decision support only. Not a medical device.
