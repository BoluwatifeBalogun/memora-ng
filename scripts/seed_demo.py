"""
Fill a fresh installation with SAMPLE patients and SYNTHETIC phantom studies so
the dashboard, records and reports can be demonstrated before real use.
The names are invented and the slices are drawn geometry, not MRI data.

    python scripts/seed_demo.py            # adds the sample data
    python scripts/seed_demo.py --clear    # removes every patient and study
"""
import argparse
import io
import os
import sys
from datetime import timedelta

import numpy as np
from werkzeug.datastructures import FileStorage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from make_phantoms import phantom_slice  # noqa: E402

from memora import create_app  # noqa: E402
from memora.db import Patient, Prediction, User, db, now  # noqa: E402
from memora.study_service import delete_study_files, run_study  # noqa: E402

PEOPLE = [
    ("Ngozi Eze", 71, "Female", "Enugu", 1.55), ("Ibrahim Musa", 64, "Male", "Kano", 1.0), ("Folake Adeyemi", 78, "Female", "Lagos", 1.9),
    ("Emeka Nwosu", 69, "Male", "Anambra", 1.3), ("Zainab Abubakar", 73, "Female", "Kaduna", 1.7), ("Tunde Bakare", 58, "Male", "Oyo", 0.9),
    ("Blessing Okon", 66, "Female", "Akwa Ibom", 1.2), ("Yakubu Danjuma", 81, "Male", "Plateau", 2.0), ("Aisha Lawal", 62, "Female", "FCT Abuja", 0.95),
    ("Ebiere Tamuno", 75, "Female", "Rivers", 1.45), ("Garba Shehu", 70, "Male", "Borno", 1.1), ("Kemi Ogunleye", 68, "Female", "Ogun", 1.6),
]


def files(vent, seed, n=18):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        buf = io.BytesIO()
        phantom_slice(0.2 + 0.6 * i / (n - 1), vent, rng).save(buf, "PNG")
        buf.seek(0)
        out.append(FileStorage(buf, filename=f"phantom_{i + 1:02d}.png"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--email", default="clinician@memora.ng")
    a = ap.parse_args()
    app = create_app()
    with app.app_context():
        if a.clear:
            for p in Patient.query.all():
                for img in p.images.all():
                    delete_study_files(img)
                db.session.delete(p)
            db.session.commit()
            print("cleared")
            sys.exit()
        user = User.query.filter_by(Email=a.email).first() or sys.exit(f"No user {a.email}")
        rng = np.random.default_rng(11)
        for i, (name, age, gender, state, vent) in enumerate(PEOPLE):
            p = Patient(User_ID=user.User_ID, Full_Name=name, Age=age, Gender=gender, State=state, Hospital_No=f"SAMPLE-{1000 + i}",
                        Phone_Number="", Address="Sample record")
            db.session.add(p)
            db.session.commit()
            for k in range(2 if i % 4 == 0 else 1):
                pred, _w = run_study(p, files(vent + 0.15 * k, seed=100 + i * 7 + k), None, user)
                pred.Prediction_Date = now() - timedelta(days=int(rng.integers(0, 13)), hours=int(rng.integers(0, 9)))
                db.session.commit()
        print("added", len(PEOPLE), "sample patients and", Prediction.query.count(), "studies for", user.Email)
