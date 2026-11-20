import io
import os
import re
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from make_phantoms import phantom_slice  # noqa: E402

from memora import create_app  # noqa: E402
from memora.ai import engine as engine_mod  # noqa: E402


@pytest.fixture()
def app(tmp_path):
    engine_mod._engine = None
    application = create_app({"INSTANCE_DIR": str(tmp_path), "TESTING": True, "MODEL_STORE": str(tmp_path / "store")})
    yield application
    engine_mod._engine = None


@pytest.fixture()
def client(app):
    return app.test_client()


def token(client, path="/sign-in"):
    html = client.get(path).get_data(as_text=True)
    return re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)


def login(client, email="clinician@memora.ng", password="Clinic@2026"):
    return client.post("/sign-in", data={"csrf_token": token(client), "email": email, "password": password})


def post(client, path, data=None, page=None, **kw):
    data = dict(data or {})
    data["csrf_token"] = token(client, page or "/app/account")
    return client.post(path, data=data, **kw)


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def phantom_files(n=16, vent=1.0, seed=3):
    rng = np.random.default_rng(seed)
    return [(io.BytesIO(png_bytes(phantom_slice(0.2 + 0.6 * i / max(n - 1, 1), vent, rng))), f"slice_{i + 1:02d}.png") for i in range(n)]


def make_patient(client, name="Chinedu Balogun", state="Lagos"):
    r = post(client, "/app/patients/new", {"full_name": name, "age": "67", "gender": "Male", "state": state, "phone_number": "0803 123 4567"})
    assert r.status_code == 302, r.get_data(as_text=True)[:400]
    return int(r.headers["Location"].rstrip("/").split("/")[-1])


def analyse(client, pid, files, ajax=True):
    data = {"csrf_token": token(client, "/app/account"), "patient": str(pid), "slices": files}
    headers = {"X-Requested-With": "fetch"} if ajax else {}
    return client.post("/app/analyse", data=data, content_type="multipart/form-data", headers=headers)
