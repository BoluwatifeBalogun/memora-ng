"""
Database models. Table and column names follow section 3.12.8 of the write-up:
Users, Patients, MRI_Images and Predictions. A few supporting columns and an
activity log are added for the features described in Chapter 3.
"""
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(db.Model):
    __tablename__ = "Users"
    User_ID = db.Column(db.Integer, primary_key=True)
    Full_Name = db.Column(db.String(120), nullable=False)
    Email = db.Column(db.String(160), unique=True, nullable=False, index=True)
    Password = db.Column(db.String(255), nullable=False)          # salted hash, never plain text
    Role = db.Column(db.String(20), nullable=False, default="clinician")   # admin | clinician
    Date_Created = db.Column(db.DateTime, default=now, nullable=False)
    Facility = db.Column(db.String(160))
    State = db.Column(db.String(40))
    Is_Active = db.Column(db.Boolean, default=True, nullable=False)
    Last_Login = db.Column(db.DateTime)
    patients = db.relationship("Patient", backref="owner", lazy="dynamic")

    @property
    def is_admin(self):
        return self.Role == "admin"

    @property
    def initials(self):
        parts = [p for p in self.Full_Name.replace("Dr.", "").replace("Dr ", "").split() if p]
        return "".join(p[0] for p in parts[:2]).upper() or "U"


class Patient(db.Model):
    __tablename__ = "Patients"
    Patient_ID = db.Column(db.Integer, primary_key=True)
    User_ID = db.Column(db.Integer, db.ForeignKey("Users.User_ID"), nullable=False, index=True)
    Hospital_No = db.Column(db.String(40))
    Full_Name = db.Column(db.String(120), nullable=False)
    Age = db.Column(db.Integer, nullable=False)
    Gender = db.Column(db.String(10), nullable=False)
    Phone_Number = db.Column(db.String(30))
    Address = db.Column(db.String(255))
    State = db.Column(db.String(40))
    Date_Created = db.Column(db.DateTime, default=now, nullable=False)
    images = db.relationship("MRIImage", backref="patient", lazy="dynamic", cascade="all, delete-orphan")
    predictions = db.relationship("Prediction", backref="patient", lazy="dynamic", cascade="all, delete-orphan")

    @property
    def code(self):
        return f"MEM-{self.Patient_ID:05d}"


class MRIImage(db.Model):
    """One uploaded MRI series (an ordered set of slices) for a patient."""
    __tablename__ = "MRI_Images"
    Image_ID = db.Column(db.Integer, primary_key=True)
    Patient_ID = db.Column(db.Integer, db.ForeignKey("Patients.Patient_ID"), nullable=False, index=True)
    Image_Name = db.Column(db.String(200), nullable=False)
    Upload_Date = db.Column(db.DateTime, default=now, nullable=False)
    Image_Path = db.Column(db.String(255), nullable=False)        # folder holding the prepared slices
    Processing_Status = db.Column(db.String(20), default="uploaded", nullable=False)
    Slice_Count = db.Column(db.Integer, default=0)
    Source_Kind = db.Column(db.String(20), default="images")      # images | nifti
    Exam_Date = db.Column(db.Date)
    prediction = db.relationship("Prediction", backref="image", uselist=False, cascade="all, delete-orphan")


class Prediction(db.Model):
    __tablename__ = "Predictions"
    Prediction_ID = db.Column(db.Integer, primary_key=True)
    Patient_ID = db.Column(db.Integer, db.ForeignKey("Patients.Patient_ID"), nullable=False, index=True)
    Image_ID = db.Column(db.Integer, db.ForeignKey("MRI_Images.Image_ID"), nullable=False, unique=True)
    User_ID = db.Column(db.Integer, db.ForeignKey("Users.User_ID"), nullable=False)
    Disease_Category = db.Column(db.String(60), nullable=False)
    Confidence_Score = db.Column(db.Float, nullable=False)
    Prediction_Date = db.Column(db.DateTime, default=now, nullable=False, index=True)
    Probabilities = db.Column(db.Text)       # JSON {class name: probability}
    Slice_Attention = db.Column(db.Text)     # JSON [16 weights]
    Top_Slices = db.Column(db.Text)          # JSON [slice indices with saved attention overlays]
    Series_Quality = db.Column(db.String(10), default="full")
    Engine = db.Column(db.String(10), default="demo")             # model | demo
    Model_Version = db.Column(db.String(40))
    Latency_Ms = db.Column(db.Integer)
    Clinician_Note = db.Column(db.Text)
    analyst = db.relationship("User")

    @property
    def ref(self):
        return f"RPT-{self.Prediction_ID:06d}"


class ActivityLog(db.Model):
    __tablename__ = "Activity_Log"
    Log_ID = db.Column(db.Integer, primary_key=True)
    User_ID = db.Column(db.Integer, db.ForeignKey("Users.User_ID"))
    Action = db.Column(db.String(40), nullable=False)
    Detail = db.Column(db.String(255))
    IP = db.Column(db.String(45))
    Created_At = db.Column(db.DateTime, default=now, nullable=False, index=True)
    user = db.relationship("User")
