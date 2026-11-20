"""Memora NG application factory."""
import json
import os
from datetime import datetime

from flask import Flask, g, render_template, request
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash

from . import config, security
from .ai.engine import load_engine
from .ai.labels import BY_NAME, CLASSES
from .db import User, db
from .nigeria import STATES, ZONES


def create_app(overrides: dict | None = None) -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    instance_dir = (overrides or {}).get("INSTANCE_DIR") or os.environ.get("MEMORA_INSTANCE") or app.instance_path
    app.config.update(config.build(instance_dir))
    if overrides:
        app.config.update(overrides)
    os.makedirs(app.config["UPLOAD_DIR"], exist_ok=True)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
    db.init_app(app)

    from .blueprints.admin import bp as admin_bp
    from .blueprints.auth import bp as auth_bp
    from .blueprints.public import bp as public_bp
    from .blueprints.workspace import bp as workspace_bp

    for bp in (public_bp, auth_bp, workspace_bp, admin_bp):
        app.register_blueprint(bp)

    @app.before_request
    def _before():
        security.load_user()
        security.check_csrf()

    app.after_request(security.security_headers)

    @app.context_processor
    def _inject():
        engine = load_engine(app.config["MODEL_STORE"])
        return {"csrf_token": security.csrf_token, "user": g.get("user"), "engine_mode": engine.mode,
                "CLASSES": CLASSES, "BY_NAME": BY_NAME, "STATES": STATES, "ZONES": ZONES, "year": datetime.now().year}

    @app.template_filter("pct")
    def _pct(v, digits=1):
        return f"{float(v) * 100:.{digits}f}%" if v is not None else "n/a"

    @app.template_filter("dt")
    def _dt(v, fmt="%d %b %Y, %H:%M"):
        return v.strftime(fmt) if v else ""

    @app.template_filter("fromjson")
    def _fromjson(v):
        return json.loads(v) if v else None

    @app.template_filter("tone")
    def _tone(name):
        return BY_NAME.get(name, {}).get("tone", "calm")

    def _error(code, title, message):
        tpl = "partials/error.html"
        return render_template(tpl, code=code, title=title, message=message), code

    @app.errorhandler(400)
    def _400(e):
        return _error(400, "That request could not be processed", getattr(e, "description", ""))

    @app.errorhandler(403)
    def _403(e):
        return _error(403, "You do not have access to this page", "This area is limited to administrators or to the clinician who owns the record.")

    @app.errorhandler(404)
    def _404(e):
        return _error(404, "Page not found", "The page or record you asked for does not exist.")

    @app.errorhandler(413)
    def _413(e):
        if request.headers.get("X-Requested-With") == "fetch":
            return {"ok": False, "error": "The upload is too large for this server."}, 413
        return _error(413, "Upload too large", "Reduce the number or size of the files and try again.")

    @app.errorhandler(500)
    def _500(e):
        return _error(500, "Something went wrong", "The error has been logged. Try again in a moment.")

    with app.app_context():
        db.create_all()
        _seed(app)
    return app


def _seed(app):
    """Create the first administrator (and a sample clinician outside production)."""
    if User.query.count():
        return
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@memora.ng").lower()
    admin_pw = os.environ.get("ADMIN_PASSWORD", "Memora@2026")
    db.session.add(User(Full_Name="System Administrator", Email=admin_email, Password=generate_password_hash(admin_pw),
                        Role="admin", Facility="Memora NG", State="FCT Abuja"))
    if not app.config["PRODUCTION"]:
        db.session.add(User(Full_Name="Dr. Amina Okafor", Email="clinician@memora.ng", Password=generate_password_hash("Clinic@2026"),
                            Role="clinician", Facility="Neurology Unit", State="Lagos"))
    db.session.commit()
