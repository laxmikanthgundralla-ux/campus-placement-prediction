import os

import joblib
import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "placement_model.pkl"
)

FEATURES = [
    "CGPA",
    "Backlogs",
    "Internships",
    "Projects",
    "AptitudeScore",
    "CommunicationScore",
    "Certifications",
]

# Load the trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Trained model not found. Run: python train_model.py"
    )

saved_model = joblib.load(MODEL_PATH)
model = saved_model["model"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read student information
        cgpa = float(request.form.get("cgpa", ""))
        backlogs = int(request.form.get("backlogs", ""))
        internships = int(request.form.get("internships", ""))
        projects = int(request.form.get("projects", ""))
        aptitude = float(request.form.get("aptitude", ""))
        communication = float(
            request.form.get("communication", "")
        )
        certifications = int(
            request.form.get("certifications", "")
        )

        # Validate input ranges
        if not 0 <= cgpa <= 10:
            raise ValueError("CGPA must be between 0 and 10.")

        if backlogs < 0:
            raise ValueError("Backlogs cannot be negative.")

        if internships < 0:
            raise ValueError("Internships cannot be negative.")

        if projects < 0:
            raise ValueError("Projects cannot be negative.")

        if not 0 <= aptitude <= 100:
            raise ValueError(
                "Aptitude score must be between 0 and 100."
            )

        if not 0 <= communication <= 100:
            raise ValueError(
                "Communication score must be between 0 and 100."
            )

        if certifications < 0:
            raise ValueError(
                "Certifications cannot be negative."
            )

        # Prepare input in the same order as model training
        student_data = pd.DataFrame(
            [[
                cgpa,
                backlogs,
                internships,
                projects,
                aptitude,
                communication,
                certifications,
            ]],
            columns=FEATURES,
        )

        # Predict placement class and probability
        prediction = int(model.predict(student_data)[0])

        probabilities = model.predict_proba(student_data)[0]
        classes = list(model.classes_)

        if 1 in classes:
            placement_probability = float(
                probabilities[classes.index(1)]
            ) * 100
        else:
            placement_probability = 0.0

        # Generate readiness category
        if placement_probability >= 75:
            readiness = "High Readiness"
            recommendation = (
                "Continue practising interviews and applying "
                "for suitable campus opportunities."
            )
        elif placement_probability >= 45:
            readiness = "Moderate Readiness"
            recommendation = (
                "Strengthen aptitude practice, communication, "
                "and project experience."
            )
        else:
            readiness = "Needs Improvement"
            recommendation = (
                "Focus on core subjects, aptitude preparation, "
                "projects, and internship opportunities."
            )

        result = {
            "prediction": (
                "Placement Likely"
                if prediction == 1
                else "Placement Not Likely"
            ),
            "probability": round(placement_probability, 1),
            "readiness": readiness,
            "recommendation": recommendation,
        }

        return render_template("result.html", result=result)

    except (ValueError, TypeError) as error:
        return render_template(
            "index.html",
            error=str(error),
        ), 400

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
