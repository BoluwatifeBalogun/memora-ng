"""Clinician workspace: dashboard, patients, analysis, records, model page."""
import csv
import io
import json
import os
from collections import Counter
from datetime import date, datetime, timedelta

from flask import (Blueprint, Response, abort, current_app, flash, g, redirect, render_template, request,
                   send_from_directory, url_for)
from sqlalchemy import func, or_
from werkzeug.security import check_password_hash, generate_password_hash

from ..ai.engine import load_engine
from ..ai.labels import CLASS_NAMES
from ..ai.preprocess import SEQ_LEN
from ..db import MRIImage, Patient, Prediction, db, now
from ..nigeria import STATES, ZONE_OF, ZONES
from ..security import audit, login_required
from ..study_service import StudyError, delete_study_files, run_study

bp = Blueprint("workspace", __name__, url_prefix="/app")
PER_PAGE = 12


# ---- scoping helpers: a clinician sees only their own patients, an admin sees all ----
def patients_q():
    q = Patient.query
    return q if g.user.is_admin else q.filter(Patient.User_ID == g.user.User_ID)


def predictions_q():
    q = Prediction.query.join(Patient, Prediction.Patient_ID == Patient.Patient_ID)
    return q if g.user.is_admin else q.filter(Patient.User_ID == g.user.User_ID)


def get_patient(pid: int) -> Patient:
    p = db.session.get(Patient, pid) or abort(404)
    if not g.user.is_admin and p.User_ID != g.user.User_ID:
        abort(403)
    return p


def get_prediction(sid: int) -> Prediction:
    pr = db.session.get(Prediction, sid) or abort(404)
    if not g.user.is_admin and pr.patient.User_ID != g.user.User_ID:
        abort(403)
    return pr


def wants_json() -> bool:
    return request.headers.get("X-Requested-With") == "fetch"


# ---- dashboard ---------------------------------------------------------------
@bp.route("/")
@login_required
def dashboard():
    preds = predictions_q()
    total = preds.count()
    week_ago = now() - timedelta(days=7)
    by_class = dict(preds.with_entities(Prediction.Disease_Category, func.count()).group_by(Prediction.Disease_Category).all())
    avg_conf = preds.with_entities(func.avg(Prediction.Confidence_Score)).scalar()
    flagged = total - by_class.get(CLASS_NAMES[0], 0)

    zone_counts = Counter()
    for state, n in preds.with_entities(Patient.State, func.count()).group_by(Patient.State).all():
        zone_counts[ZONE_OF.get(state, "Unspecified")] += n
    zones = [{"name": z, "count": zone_counts.get(z, 0)} for z in ZONES]
    zone_max = max([z["count"] for z in zones] + [1])

    start = (now() - timedelta(days=13)).replace(hour=0, minute=0, second=0, microsecond=0)
    daily = Counter(d.date() for (d,) in preds.filter(Prediction.Prediction_Date >= start).with_entities(Prediction.Prediction_Date).all())
    days = [start.date() + timedelta(days=i) for i in range(14)]
    trend = [{"label": d.strftime("%d %b"), "count": daily.get(d, 0)} for d in days]

    return render_template(
        "app/dashboard.html", active="home",
        stats={"patients": patients_q().count(), "studies": total, "week": preds.filter(Prediction.Prediction_Date >= week_ago).count(),
               "flagged": flagged, "avg_conf": avg_conf},
        dist=[{"name": n, "count": by_class.get(n, 0)} for n in CLASS_NAMES], total=total,
        zones=zones, zone_max=zone_max, trend=trend, trend_max=max([t["count"] for t in trend] + [1]),
        recent=preds.order_by(Prediction.Prediction_Date.desc()).limit(6).all(),
    )


# ---- patients ----------------------------------------------------------------
def _patient_form(p: Patient | None):
    f = request.form
    errors = {}
    name = (f.get("full_name") or "").strip()
    if len(name) < 3:
        errors["full_name"] = "Enter the patient's full name."
    try:
        age = int(f.get("age") or "")
        if not 1 <= age <= 120:
            raise ValueError
    except ValueError:
        age, errors["age"] = None, "Enter an age between 1 and 120."
    gender = f.get("gender")
    if gender not in ("Female", "Male"):
        errors["gender"] = "Choose a gender."
    state = f.get("state") or ""
    if state not in STATES:
        errors["state"] = "Choose the patient's state of residence."
    phone = (f.get("phone_number") or "").strip()
    if phone and not (7 <= len(phone.replace(" ", "").replace("+", "")) <= 15 and phone.replace(" ", "").replace("+", "").isdigit()):
        errors["phone_number"] = "Use digits only, for example 0803 123 4567."
    data = {"Full_Name": name[:120], "Age": age, "Gender": gender, "State": state, "Phone_Number": phone[:30],
            "Address": (f.get("address") or "").strip()[:255], "Hospital_No": (f.get("hospital_no") or "").strip()[:40]}
    return data, errors


@bp.route("/patients")
@login_required
def patients():
    q = (request.args.get("q") or "").strip()
    query = patients_q()
    if q:
        like = f"%{q}%"
        query = query.filter(or_(Patient.Full_Name.ilike(like), Patient.Hospital_No.ilike(like), Patient.State.ilike(like)))
    page = query.order_by(Patient.Date_Created.desc()).paginate(page=request.args.get("page", 1, type=int), per_page=PER_PAGE, error_out=False)
    latest = {}
    ids = [p.Patient_ID for p in page.items]
    if ids:
        for pr in Prediction.query.filter(Prediction.Patient_ID.in_(ids)).order_by(Prediction.Prediction_Date.asc()).all():
            latest[pr.Patient_ID] = pr
    return render_template("app/patients.html", active="patients", page=page, q=q, latest=latest)


@bp.route("/patients/new", methods=["GET", "POST"])
@login_required
def patient_new():
    data, errors = {}, {}
    if request.method == "POST":
        data, errors = _patient_form(None)
        if not errors:
            p = Patient(User_ID=g.user.User_ID, **data)
            db.session.add(p)
            db.session.commit()
            audit("patient_created", f"{p.code} {p.Full_Name}")
            flash("Patient record created.", "success")
            if request.form.get("then") == "analyse":
                return redirect(url_for("workspace.analyse", patient=p.Patient_ID))
            return redirect(url_for("workspace.patient_detail", pid=p.Patient_ID))
    return render_template("app/patient_form.html", active="patients", data=data, errors=errors, patient=None), (400 if errors else 200)


@bp.route("/patients/<int:pid>")
@login_required
def patient_detail(pid):
    p = get_patient(pid)
    studies = p.predictions.order_by(Prediction.Prediction_Date.desc()).all()
    return render_template("app/patient_detail.html", active="patients", patient=p, studies=studies)


@bp.route("/patients/<int:pid>/edit", methods=["GET", "POST"])
@login_required
def patient_edit(pid):
    p = get_patient(pid)
    data = {c: getattr(p, c) for c in ("Full_Name", "Age", "Gender", "State", "Phone_Number", "Address", "Hospital_No")}
    errors = {}
    if request.method == "POST":
        data, errors = _patient_form(p)
        if not errors:
            for k, v in data.items():
                setattr(p, k, v)
            db.session.commit()
            audit("patient_updated", p.code)
            flash("Patient record updated.", "success")
            return redirect(url_for("workspace.patient_detail", pid=pid))
    return render_template("app/patient_form.html", active="patients", data=data, errors=errors, patient=p), (400 if errors else 200)


@bp.route("/patients/<int:pid>/delete", methods=["POST"])
@login_required
def patient_delete(pid):
    p = get_patient(pid)
    for img in p.images.all():
        delete_study_files(img)
    code = p.code
    db.session.delete(p)
    db.session.commit()
    audit("patient_deleted", code)
    flash("Patient record and its studies were deleted.", "info")
    return redirect(url_for("workspace.patients"))


# ---- analysis ----------------------------------------------------------------
@bp.route("/analyse", methods=["GET", "POST"])
@login_required
def analyse():
    mine = Patient.query.filter(Patient.User_ID == g.user.User_ID).order_by(Patient.Full_Name).all()
    selected = request.values.get("patient", type=int)
    if request.method == "POST":
        try:
            patient = get_patient(selected) if selected else None
            if patient is None:
                raise StudyError("Choose the patient this scan belongs to.")
            exam = None
            if request.form.get("exam_date"):
                try:
                    exam = datetime.strptime(request.form["exam_date"], "%Y-%m-%d").date()
                except ValueError:
                    raise StudyError("The examination date is not valid.")
                if exam > date.today():
                    raise StudyError("The examination date cannot be in the future.")
            pred, warnings = run_study(patient, request.files.getlist("slices"), exam, g.user)
        except StudyError as exc:
            if wants_json():
                return {"ok": False, "error": str(exc)}, 422
            flash(str(exc), "error")
            return render_template("app/analyse.html", active="analyse", patients=mine, selected=selected, seq_len=SEQ_LEN), 422
        audit("study_analysed", f"{pred.ref} {pred.Disease_Category} {pred.Confidence_Score:.2f} [{pred.Engine}]")
        for w in warnings:
            flash(w, "warn")
        target = url_for("workspace.study", sid=pred.Prediction_ID)
        return {"ok": True, "redirect": target} if wants_json() else redirect(target)
    return render_template("app/analyse.html", active="analyse", patients=mine, selected=selected, seq_len=SEQ_LEN)


# ---- records -----------------------------------------------------------------
def _filtered():
    q = (request.args.get("q") or "").strip()
    cat = request.args.get("category") or ""
    query = predictions_q()
    if q:
        like = f"%{q}%"
        query = query.filter(or_(Patient.Full_Name.ilike(like), Patient.Hospital_No.ilike(like), Patient.State.ilike(like)))
    if cat in CLASS_NAMES:
        query = query.filter(Prediction.Disease_Category == cat)
    return query.order_by(Prediction.Prediction_Date.desc()), q, cat


@bp.route("/studies")
@login_required
def studies():
    query, q, cat = _filtered()
    page = query.paginate(page=request.args.get("page", 1, type=int), per_page=PER_PAGE, error_out=False)
    return render_template("app/studies.html", active="records", page=page, q=q, category=cat)


@bp.route("/studies/export.csv")
@login_required
def studies_export():
    query, _q, _cat = _filtered()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Report", "Date", "Patient code", "Patient", "Age", "Gender", "State", "Zone", "Category", "Confidence", "Slices", "Engine", "Model version", "Analysed by"])
    safe = lambda s: ("'" + s) if s and s[0] in "=+-@" else (s or "")   # stop spreadsheet formula injection  # noqa: E731
    for pr in query.all():
        p = pr.patient
        w.writerow([pr.ref, pr.Prediction_Date.strftime("%Y-%m-%d %H:%M"), p.code, safe(p.Full_Name), p.Age, p.Gender, p.State or "",
                    ZONE_OF.get(p.State, ""), pr.Disease_Category, f"{pr.Confidence_Score:.4f}", pr.image.Slice_Count, pr.Engine,
                    pr.Model_Version or "", safe(pr.analyst.Full_Name)])
    audit("records_exported", "CSV export")
    return Response(buf.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=memora_records.csv"})


def _study_context(pr: Prediction):
    probs = json.loads(pr.Probabilities or "{}")
    att = json.loads(pr.Slice_Attention or "[]")
    top = json.loads(pr.Top_Slices or "[]")
    peak = max(att) if att else 1
    slices = [{"i": i, "weight": w, "rel": (w / peak if peak else 0), "top": i in top} for i, w in enumerate(att)]
    return {"pr": pr, "patient": pr.patient, "image": pr.image, "top": top, "slices": slices,
            "probs": [{"name": n, "p": probs.get(n, 0.0)} for n in CLASS_NAMES]}


@bp.route("/studies/<int:sid>")
@login_required
def study(sid):
    return render_template("app/study.html", active="records", **_study_context(get_prediction(sid)))


@bp.route("/studies/<int:sid>/report")
@login_required
def study_report(sid):
    return render_template("app/report.html", **_study_context(get_prediction(sid)))


@bp.route("/studies/<int:sid>/note", methods=["POST"])
@login_required
def study_note(sid):
    pr = get_prediction(sid)
    pr.Clinician_Note = (request.form.get("note") or "").strip()[:2000]
    db.session.commit()
    audit("note_saved", pr.ref)
    flash("Clinical note saved.", "success")
    return redirect(url_for("workspace.study", sid=sid))


@bp.route("/studies/<int:sid>/delete", methods=["POST"])
@login_required
def study_delete(sid):
    pr = get_prediction(sid)
    pid, ref, img = pr.Patient_ID, pr.ref, pr.image
    delete_study_files(img)
    db.session.delete(img)          # cascades to the prediction
    db.session.commit()
    audit("study_deleted", ref)
    flash("Study deleted.", "info")
    return redirect(url_for("workspace.patient_detail", pid=pid))


@bp.route("/media/<int:sid>/<name>")
@login_required
def media(sid, name):
    pr = get_prediction(sid)
    if not (name.endswith(".png") and (name.startswith("slice_") or name.startswith("overlay_"))):
        abort(404)
    folder = os.path.join(current_app.config["UPLOAD_DIR"], os.path.basename(pr.image.Image_Path))
    return send_from_directory(folder, name, max_age=3600)


# ---- model and account -------------------------------------------------------
@bp.route("/model")
@login_required
def model():
    engine = load_engine(current_app.config["MODEL_STORE"])
    return render_template("app/model.html", active="model", card=engine.card or {}, seq_len=SEQ_LEN,
                           load_error=getattr(engine, "load_error", None))


@bp.route("/account", methods=["GET", "POST"])
@login_required
def account():
    errors = {}
    if request.method == "POST":
        cur, new, again = (request.form.get(k) or "" for k in ("current", "new", "again"))
        if not check_password_hash(g.user.Password, cur):
            errors["current"] = "Current password is incorrect."
        if len(new) < 8 or new.isalpha() or new.isdigit():
            errors["new"] = "Use at least 8 characters with a mix of letters and numbers or symbols."
        elif new != again:
            errors["again"] = "The two new passwords do not match."
        if not errors:
            g.user.Password = generate_password_hash(new)
            db.session.commit()
            audit("password_changed", "Changed own password")
            flash("Password updated.", "success")
            return redirect(url_for("workspace.account"))
    return render_template("app/account.html", active="", errors=errors), (400 if errors else 200)
