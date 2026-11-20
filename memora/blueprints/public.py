from flask import Blueprint, current_app, render_template

from ..ai.engine import load_engine

bp = Blueprint("public", __name__)


@bp.route("/")
def home():
    return render_template("public/home.html")


@bp.route("/healthz")
def healthz():
    engine = load_engine(current_app.config["MODEL_STORE"])
    return {"status": "ok", "engine": engine.mode, "model_version": engine.version}
