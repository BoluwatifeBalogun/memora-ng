"""End-to-end tests. Each maps to a use case or activity step in Chapter 3."""
import io
import os

import numpy as np
import pytest
from PIL import Image

from conftest import analyse, login, make_patient, phantom_files, png_bytes, post, token
from memora.ai import engine as engine_mod
from memora.ai import preprocess as pp
from memora.ai.labels import CLASS_NAMES
from memora.db import ActivityLog, MRIImage, Patient, Prediction, User


# ---- authentication and roles --------------------------------------------------
def test_workspace_requires_login(client):
    r = client.get("/app/")
    assert r.status_code == 302 and "/sign-in" in r.headers["Location"]


def test_login_wrong_then_right(client):
    assert b"incorrect" in login(client, password="nope").data
    r = login(client)
    assert r.status_code == 302 and r.headers["Location"].endswith("/app/")
    assert client.get("/app/").status_code == 200


def test_login_is_throttled(client):
    for _ in range(5):
        login(client, email="x@y.ng", password="bad")
    assert login(client, email="x@y.ng", password="bad").status_code == 429


def test_post_without_csrf_is_rejected(client):
    login(client)
    assert client.post("/app/patients/new", data={"full_name": "A B C"}).status_code == 400


def test_open_redirect_is_blocked(client):
    r = client.post("/sign-in?next=//evil.example", data={"csrf_token": token(client), "email": "clinician@memora.ng", "password": "Clinic@2026"})
    assert r.headers["Location"].endswith("/app/")


def test_clinician_cannot_open_admin(client):
    login(client)
    assert client.get("/app/admin/users").status_code == 403


def test_admin_registers_user_who_can_sign_in(client, app):
    login(client, "admin@memora.ng", "Memora@2026")
    r = post(client, "/app/admin/users", {"full_name": "Dr. Hauwa Bello", "email": "hauwa@uch.ng", "role": "clinician", "facility": "UCH", "state": "Oyo", "password": "Welcome#24"})
    assert r.status_code == 302
    post(client, "/sign-out")
    assert login(client, "hauwa@uch.ng", "Welcome#24").status_code == 302
    with app.app_context():
        assert not User.query.filter_by(Email="hauwa@uch.ng").first().Password.startswith("Welcome")   # hashed


def test_admin_rejects_duplicate_and_weak_password(client):
    login(client, "admin@memora.ng", "Memora@2026")
    r = post(client, "/app/admin/users", {"full_name": "Dup", "email": "clinician@memora.ng", "role": "clinician", "password": "short"})
    assert r.status_code == 400 and b"already exists" in r.data and b"at least 8" in r.data


def test_deactivated_user_cannot_sign_in(client, app):
    login(client, "admin@memora.ng", "Memora@2026")
    with app.app_context():
        uid = User.query.filter_by(Email="clinician@memora.ng").first().User_ID
    post(client, f"/app/admin/users/{uid}/toggle")
    post(client, "/sign-out")
    assert b"deactivated" in login(client).data


# ---- patients ---------------------------------------------------------------------
def test_patient_validation_and_create(client, app):
    login(client)
    r = post(client, "/app/patients/new", {"full_name": "A", "age": "400", "gender": "", "state": "Atlantis"})
    assert r.status_code == 400 and r.data.count(b'class="err"') >= 4
    pid = make_patient(client)
    assert client.get(f"/app/patients/{pid}").status_code == 200
    with app.app_context():
        assert Patient.query.count() == 1


def test_clinician_cannot_see_another_clinicians_patient(client):
    login(client)
    pid = make_patient(client)
    post(client, "/sign-out")
    login(client, "admin@memora.ng", "Memora@2026")
    post(client, "/app/admin/users", {"full_name": "Dr. Second", "email": "two@x.ng", "role": "clinician", "password": "Welcome#24"})
    assert client.get(f"/app/patients/{pid}").status_code == 200          # admin may view
    post(client, "/sign-out")
    login(client, "two@x.ng", "Welcome#24")
    assert client.get(f"/app/patients/{pid}").status_code == 403
    assert analyse(client, pid, phantom_files(4)).status_code == 403


# ---- upload, validation, prediction, storage ---------------------------------------
def test_full_series_is_analysed_and_stored(client, app):
    login(client)
    pid = make_patient(client)
    r = analyse(client, pid, phantom_files(20, vent=1.8))
    assert r.status_code == 200 and r.json["ok"], r.get_data(as_text=True)
    page = client.get(r.json["redirect"])
    assert page.status_code == 200 and b"Demonstration result" in page.data
    with app.app_context():
        pr = Prediction.query.one()
        img = MRIImage.query.one()
        assert pr.Disease_Category in CLASS_NAMES and 0 < pr.Confidence_Score <= 1
        assert img.Processing_Status == "processed" and img.Slice_Count == 20 and pr.Series_Quality == "full"
        folder = os.path.join(app.config["UPLOAD_DIR"], img.Image_Path)
        assert len([f for f in os.listdir(folder) if f.startswith("slice_")]) == 16
        assert len([f for f in os.listdir(folder) if f.startswith("overlay_")]) == 3
        assert ActivityLog.query.filter_by(Action="study_analysed").count() == 1
        sid = pr.Prediction_ID
    assert client.get(f"/app/media/{sid}/slice_00.png").mimetype == "image/png"
    assert client.get(f"/app/media/{sid}/../../secret.key").status_code == 404
    assert client.get(f"/app/studies/{sid}/report").status_code == 200
    assert b"Chinedu Balogun" in client.get("/app/studies/export.csv").data


def test_demo_rule_orders_ventricle_sizes(client, app):
    """Bigger ventricles must not look more normal than smaller ones (sanity of the placeholder)."""
    login(client)
    pid = make_patient(client)
    analyse(client, pid, phantom_files(16, vent=0.9))
    analyse(client, pid, phantom_files(16, vent=1.9))
    with app.app_context():
        a, b = Prediction.query.order_by(Prediction.Prediction_ID).all()
        import json
        assert json.loads(a.Probabilities)[CLASS_NAMES[2]] < json.loads(b.Probabilities)[CLASS_NAMES[2]]


def test_single_slice_is_flagged_as_limited(client, app):
    login(client)
    pid = make_patient(client)
    r = analyse(client, pid, phantom_files(1))
    assert r.json["ok"]
    assert b"Limited context" in client.get(r.json["redirect"]).data


def test_colour_photo_is_rejected(client, app):
    login(client)
    pid = make_patient(client)
    rng = np.random.default_rng(0)
    photo = Image.fromarray(rng.integers(0, 255, (200, 200, 3), dtype=np.uint8))
    r = analyse(client, pid, [(io.BytesIO(png_bytes(photo)), "holiday.png")])
    assert r.status_code == 422 and "colour" in r.json["error"]
    with app.app_context():
        assert Prediction.query.count() == 0 and MRIImage.query.count() == 0


@pytest.mark.parametrize("name,data,needle", [
    ("notes.txt", b"hello", "not a supported file"),
    ("fake.png", b"this is not an image", "could not be read"),
    ("blank.png", png_bytes(Image.new("L", (200, 200), 0)), "blank"),
    ("page.png", png_bytes(Image.new("L", (200, 200), 240)), "blank"),
])
def test_bad_files_are_rejected(client, name, data, needle):
    login(client)
    pid = make_patient(client)
    r = analyse(client, pid, [(io.BytesIO(data), name)])
    assert r.status_code == 422 and needle in r.json["error"]


def test_no_file_and_no_patient(client):
    login(client)
    pid = make_patient(client)
    assert analyse(client, pid, []).status_code == 422
    r = client.post("/app/analyse", data={"csrf_token": token(client, "/app/account"), "slices": phantom_files(2)}, content_type="multipart/form-data", headers={"X-Requested-With": "fetch"})
    assert r.status_code == 422 and "patient" in r.json["error"].lower()


def test_nifti_volume_is_accepted(client, app, tmp_path):
    nib = pytest.importorskip("nibabel")
    rng = np.random.default_rng(5)
    vol = np.stack([np.rot90(np.asarray(__import__("make_phantoms").phantom_slice(i / 59, 1.2, rng, 128), dtype=np.float32), -1) for i in range(60)], axis=2)
    path = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(path))
    login(client)
    pid = make_patient(client)
    r = analyse(client, pid, [(io.BytesIO(path.read_bytes()), "scan.nii.gz")])
    assert r.status_code == 200 and r.json["ok"], r.get_data(as_text=True)
    with app.app_context():
        assert MRIImage.query.one().Source_Kind == "nifti"


def test_delete_study_removes_files(client, app):
    login(client)
    pid = make_patient(client)
    analyse(client, pid, phantom_files(8))
    with app.app_context():
        pr = Prediction.query.one()
        sid, folder = pr.Prediction_ID, os.path.join(app.config["UPLOAD_DIR"], pr.image.Image_Path)
    assert os.path.isdir(folder)
    assert post(client, f"/app/studies/{sid}/delete").status_code == 302
    assert not os.path.exists(folder)
    with app.app_context():
        assert Prediction.query.count() == 0 and MRIImage.query.count() == 0


def test_pages_render(client):
    login(client, "admin@memora.ng", "Memora@2026")
    for url in ["/", "/app/", "/app/patients", "/app/analyse", "/app/studies", "/app/model", "/app/account", "/app/admin/users", "/app/admin/activity"]:
        assert client.get(url).status_code == 200, url


# ---- preprocessing and the trained-model path ----------------------------------------
def test_preprocessing_units():
    assert list(pp.pick_indices(32, 16))[0] == 0 and list(pp.pick_indices(32, 16))[-1] == 31
    assert list(pp.pick_indices(1, 16)) == [0] * 16
    assert sorted(["s10.png", "s2.png", "s1.png"], key=pp.natural_key) == ["s1.png", "s2.png", "s10.png"]
    assert [pp.series_quality(n) for n in (1, 8, 16)] == ["single", "sparse", "full"]
    g = pp.prepare_slice(Image.new("RGB", (300, 180), (90, 90, 90)))
    assert g.shape == (224, 224) and 0 <= g.min() and g.max() <= 1
    assert pp.to_model_input(np.zeros((16, 224, 224), np.float32)).shape == (1, 16, 3, 224, 224)


def test_dataset_folder_mapping():
    from training.dataset import class_of
    assert [class_of(n) for n in ("Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia", "CN", "MCI", "AD", "sub-01")] == [0, 1, 2, 2, 0, 1, 2, None]


def test_trained_model_path(client, app, tmp_path):
    """Export an untrained ViT-LSTM to ONNX and check the web app serves it in model mode."""
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    import json

    from training.export_onnx import export
    from training.model import ViTLSTM

    store = app.config["MODEL_STORE"]
    os.makedirs(store, exist_ok=True)
    export(ViTLSTM(3, pretrained=False), os.path.join(store, engine_mod.MODEL_FILE))
    with open(os.path.join(store, engine_mod.CARD_FILE), "w") as fh:
        json.dump({"version": "test", "labels": CLASS_NAMES}, fh)
    engine_mod._engine = None
    assert client.get("/healthz").json["engine"] == "model"
    login(client)
    pid = make_patient(client)
    r = analyse(client, pid, phantom_files(16))
    assert r.json["ok"]
    with app.app_context():
        pr = Prediction.query.one()
        assert pr.Engine == "model" and abs(sum(json.loads(pr.Probabilities).values()) - 1) < 1e-4
        assert abs(sum(json.loads(pr.Slice_Attention)) - 1) < 1e-3
    assert b"Demonstration result" not in client.get(r.json["redirect"]).data
