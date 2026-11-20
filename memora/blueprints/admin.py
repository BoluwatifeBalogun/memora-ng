"""Administrator area: user registration and management, activity log."""
import re
import secrets

from flask import Blueprint, flash, g, redirect, render_template, request, url_for
from werkzeug.security import generate_password_hash

from ..db import ActivityLog, Patient, Prediction, User, db
from ..nigeria import STATES
from ..security import admin_required, audit

bp = Blueprint("admin", __name__, url_prefix="/app/admin")
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@bp.route("/users", methods=["GET", "POST"])
@admin_required
def users():
    data, errors = {}, {}
    if request.method == "POST":
        f = request.form
        data = {"full_name": (f.get("full_name") or "").strip(), "email": (f.get("email") or "").strip().lower(),
                "role": f.get("role") or "clinician", "facility": (f.get("facility") or "").strip(), "state": f.get("state") or ""}
        password = f.get("password") or ""
        if len(data["full_name"]) < 3:
            errors["full_name"] = "Enter the user's full name."
        if not EMAIL.match(data["email"]):
            errors["email"] = "Enter a valid email address."
        elif User.query.filter_by(Email=data["email"]).first():
            errors["email"] = "A user with this email already exists."
        if data["role"] not in ("admin", "clinician"):
            errors["role"] = "Choose a role."
        if data["state"] and data["state"] not in STATES:
            errors["state"] = "Choose a state from the list."
        if len(password) < 8 or password.isalpha() or password.isdigit():
            errors["password"] = "Use at least 8 characters with a mix of letters and numbers or symbols."
        if not errors:
            u = User(Full_Name=data["full_name"][:120], Email=data["email"][:160], Password=generate_password_hash(password),
                     Role=data["role"], Facility=data["facility"][:160], State=data["state"])
            db.session.add(u)
            db.session.commit()
            audit("user_registered", f"{u.Email} as {u.Role}")
            flash(f"{u.Full_Name} can now sign in.", "success")
            return redirect(url_for("admin.users"))
    people = User.query.order_by(User.Date_Created.desc()).all()
    counts = dict(db.session.query(Prediction.User_ID, db.func.count()).group_by(Prediction.User_ID).all())
    pcounts = dict(db.session.query(Patient.User_ID, db.func.count()).group_by(Patient.User_ID).all())
    return render_template("admin/users.html", active="", people=people, counts=counts, pcounts=pcounts,
                           data=data, errors=errors), (400 if errors else 200)


def _target(uid):
    u = db.session.get(User, uid)
    if not u:
        flash("User not found.", "error")
    return u


@bp.route("/users/<int:uid>/toggle", methods=["POST"])
@admin_required
def toggle(uid):
    u = _target(uid)
    if u:
        if u.User_ID == g.user.User_ID:
            flash("You cannot deactivate your own account.", "error")
        else:
            u.Is_Active = not u.Is_Active
            db.session.commit()
            audit("user_status", f"{u.Email} {'activated' if u.Is_Active else 'deactivated'}")
            flash(f"{u.Full_Name} is now {'active' if u.Is_Active else 'deactivated'}.", "info")
    return redirect(url_for("admin.users"))


@bp.route("/users/<int:uid>/role", methods=["POST"])
@admin_required
def role(uid):
    u = _target(uid)
    if u:
        if u.User_ID == g.user.User_ID:
            flash("You cannot change your own role.", "error")
        else:
            u.Role = "clinician" if u.Role == "admin" else "admin"
            db.session.commit()
            audit("user_role", f"{u.Email} is now {u.Role}")
            flash(f"{u.Full_Name} is now {'an administrator' if u.Role == 'admin' else 'a healthcare professional'}.", "info")
    return redirect(url_for("admin.users"))


@bp.route("/users/<int:uid>/reset", methods=["POST"])
@admin_required
def reset(uid):
    u = _target(uid)
    if u:
        temp = "Mem-" + secrets.token_urlsafe(6)
        u.Password = generate_password_hash(temp)
        db.session.commit()
        audit("password_reset", u.Email)
        flash(f"Temporary password for {u.Full_Name}: {temp}  Share it privately and ask them to change it after signing in.", "success")
    return redirect(url_for("admin.users"))


@bp.route("/activity")
@admin_required
def activity():
    page = ActivityLog.query.order_by(ActivityLog.Created_At.desc()).paginate(page=request.args.get("page", 1, type=int),
                       per_page=25, error_out=False)
    return render_template("admin/activity.html", active="", page=page)
