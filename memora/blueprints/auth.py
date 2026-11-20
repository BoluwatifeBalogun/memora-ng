from urllib.parse import urlparse

from flask import Blueprint, current_app, flash, g, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from ..db import User, db, now
from ..security import audit, clear_failures, client_ip, login_required, note_failure, throttled

bp = Blueprint("auth", __name__)

_DUMMY_HASH = generate_password_hash("not-a-real-password")


def _safe_next(target: str | None) -> str:
    if target and target.startswith("/") and not target.startswith("//") and not urlparse(target).netloc:
        return target
    return url_for("workspace.dashboard")


@bp.route("/sign-in", methods=["GET", "POST"])
def sign_in():
    if g.user:
        return redirect(url_for("workspace.dashboard"))
    email = ""
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        key = f"{client_ip()}|{email}"
        if throttled(key):
            flash("Too many failed attempts. Wait ten minutes and try again.", "error")
            return render_template("auth/sign_in.html", email=email), 429
        user = User.query.filter_by(Email=email).first()
        ok = check_password_hash(user.Password if user else _DUMMY_HASH, password)   # same work either way
        if user and ok and user.Is_Active:
            clear_failures(key)
            next_url = _safe_next(request.args.get("next"))
            session.clear()
            session["uid"] = user.User_ID
            session.permanent = True
            user.Last_Login = now()
            db.session.commit()
            audit("sign_in", "Signed in", user_id=user.User_ID)
            return redirect(next_url)
        note_failure(key)
        if user and ok and not user.Is_Active:
            flash("This account has been deactivated. Contact your administrator.", "error")
        else:
            flash("Email or password is incorrect.", "error")
        audit("sign_in_failed", email[:120], user_id=user.User_ID if user else None)
    return render_template("auth/sign_in.html", email=email, demo=current_app.config["SHOW_DEMO_LOGINS"])


@bp.route("/sign-out", methods=["POST"])
@login_required
def sign_out():
    audit("sign_out", "Signed out")
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("public.home"))
