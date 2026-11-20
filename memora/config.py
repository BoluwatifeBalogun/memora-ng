"""Application settings, read from environment variables."""
import os
import secrets

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _secret(instance_dir: str) -> str:
    env = os.environ.get("SECRET_KEY")
    if env:
        return env
    path = os.path.join(instance_dir, "secret.key")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(secrets.token_hex(32))
    with open(path, encoding="utf-8") as fh:
        return fh.read().strip()


def build(instance_dir: str) -> dict:
    os.makedirs(instance_dir, exist_ok=True)
    db_url = os.environ.get("DATABASE_URL", "").strip()
    if db_url.startswith("mysql://"):                       # hosting panels often give this form
        db_url = "mysql+pymysql://" + db_url[len("mysql://"):]
    if not db_url:
        db_url = "sqlite:///" + os.path.join(instance_dir, "memora.db")
    production = os.environ.get("MEMORA_ENV", "development").lower() == "production"
    return {
        "SECRET_KEY": _secret(instance_dir),
        "SQLALCHEMY_DATABASE_URI": db_url,
        "SQLALCHEMY_TRACK_MODIFICATIONS": False,
        "SQLALCHEMY_ENGINE_OPTIONS": {"pool_pre_ping": True, "pool_recycle": 280},
        "MAX_CONTENT_LENGTH": int(os.environ.get("MAX_UPLOAD_MB", "96")) * 1024 * 1024,
        "UPLOAD_DIR": os.environ.get("UPLOAD_DIR", os.path.join(instance_dir, "studies")),
        "MODEL_STORE": os.environ.get("MODEL_STORE", os.path.join(BASE_DIR, "model_store")),
        "MAX_SLICES": 64,
        "PRODUCTION": production,
        "SHOW_DEMO_LOGINS": os.environ.get("SHOW_DEMO_LOGINS", "0" if production else "1") == "1",
        "SESSION_COOKIE_HTTPONLY": True,
        "SESSION_COOKIE_SAMESITE": "Lax",
        "SESSION_COOKIE_SECURE": production,
        "PERMANENT_SESSION_LIFETIME": 60 * 60 * 8,
    }
