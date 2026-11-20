"""Sessions, CSRF protection, role checks, login throttling and audit logging."""
import hmac
import secrets
import time
from functools import wraps

from flask import abort, current_app, flash, g, redirect, request, session, url_for

from .db import ActivityLog, User, db

_FAILS: dict[str, list[float]] = {}
MAX_FAILS, WINDOW = 5, 600


def csrf_token() -> str:
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_urlsafe(32)
    return session["_csrf"]


def check_csrf():
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        sent = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token", "")
        if not sent or not hmac.compare_digest(sent, session.get("_csrf", "")):
            abort(400, description="Your session expired or the form was not sent from this site. Reload the page and try again.")


def load_user():
    g.user = None
    uid = session.get("uid")
    if uid:
        user = db.session.get(User, uid)
        if user and user.Is_Active:
            g.user = user
        else:
            session.clear()


def login_required(view):
    @wraps(view)
    def wrapped(*a, **kw):
        if g.user is None:
            flash("Sign in to continue.", "info")
            return redirect(url_for("auth.sign_in", next=request.path))
        return view(*a, **kw)
    return wrapped


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped(*a, **kw):
        if not g.user.is_admin:
            abort(403)
        return view(*a, **kw)
    return wrapped


def client_ip() -> str:
    fwd = request.headers.get("X-Forwarded-For", "")
    return (fwd.split(",")[0].strip() if fwd else request.remote_addr) or "unknown"


def throttled(key: str) -> bool:
    t = time.time()
    hits = [h for h in _FAILS.get(key, []) if t - h < WINDOW]
    _FAILS[key] = hits
    return len(hits) >= MAX_FAILS


def note_failure(key: str):
    _FAILS.setdefault(key, []).append(time.time())


def clear_failures(key: str):
    _FAILS.pop(key, None)


def audit(action: str, detail: str = "", user_id=None):
    try:
        uid = user_id if user_id is not None else (g.user.User_ID if getattr(g, "user", None) else None)
        db.session.add(ActivityLog(User_ID=uid, Action=action, Detail=detail[:255], IP=client_ip()))
        db.session.commit()
    except Exception:           # logging must never break a request
        db.session.rollback()
        current_app.logger.exception("audit log failed")


def security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    resp.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; font-src 'self'; "
        "script-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
    )
    return resp
