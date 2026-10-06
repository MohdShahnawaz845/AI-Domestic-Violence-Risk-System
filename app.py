from flask import Flask, render_template, request
import joblib
import pandas as pd
from datetime import datetime
import sqlite3

app = Flask(__name__)

# Temporary session history
assessment_history = []

# Load trained model
model = joblib.load("model/domestic_violence_model.pkl")


# -----------------------------------------
# Save assessment to database
# -----------------------------------------

def save_assessment(
    threats,
    physical_harm,
    controlling_behavior,
    verbal_abuse,
    isolation,
    financial_control,
    risk
):
    connection = sqlite3.connect("risk_assessments.db")
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO assessments (
            threats,
            physical_harm,
            controlling_behavior,
            verbal_abuse,
            isolation,
            financial_control,
            risk_level,
            assessment_time
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        threats,
        physical_harm,
        controlling_behavior,
        verbal_abuse,
        isolation,
        financial_control,
        risk,
        datetime.now().strftime("%d-%m-%Y %I:%M %p")
    ))

    connection.commit()
    connection.close()


# -----------------------------------------
# Home Page
# -----------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------------------
# Assessment Page
# -----------------------------------------

@app.route("/assessment")
def assessment():
    return render_template("assessment.html")


# -----------------------------------------
# Prediction
# -----------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    threats = int(request.form["threats"])
    physical_harm = int(request.form["physical_harm"])
    controlling_behavior = int(request.form["controlling_behavior"])
    verbal_abuse = int(request.form["verbal_abuse"])
    isolation = int(request.form["isolation"])
    financial_control = int(request.form["financial_control"])

    # Create input data
    input_data = pd.DataFrame(
        [[
            threats,
            physical_harm,
            controlling_behavior,
            verbal_abuse,
            isolation,
            financial_control
        ]],
        columns=[
            "threats",
            "physical_harm",
            "controlling_behavior",
            "verbal_abuse",
            "isolation",
            "financial_control"
        ]
    )

    # -----------------------------------------
    # Model Prediction
    # -----------------------------------------

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    confidence = round(max(probabilities) * 100, 1)

    prediction = int(prediction)

    # Risk level
    if prediction == 0:
        risk = "Low Risk"
    elif prediction == 1:
        risk = "Medium Risk"
    else:
        risk = "High Risk"


    # -----------------------------------------
    # Factor Contribution Calculation
    # -----------------------------------------

    feature_names = [
        "Threats",
        "Physical Harm",
        "Controlling Behavior",
        "Verbal Abuse",
        "Isolation",
        "Financial Control"
    ]

    feature_values = [
        threats,
        physical_harm,
        controlling_behavior,
        verbal_abuse,
        isolation,
        financial_control
    ]

    # Random Forest feature importance
    feature_importances = model.feature_importances_

    # Find factors selected as Yes
    selected_indexes = [
        i
        for i in range(len(feature_values))
        if feature_values[i] == 1
    ]

    # Initially every factor has 0%
    feature_percentages = {
        feature_names[i]: 0.0
        for i in range(len(feature_names))
    }

    # -----------------------------------------
    # Distribute Model Confidence
    # among selected Yes factors
    # -----------------------------------------

    if selected_indexes:

        selected_importance = sum(
            feature_importances[i]
            for i in selected_indexes
        )

        calculated_percentages = []

        for i in selected_indexes:

            if selected_importance > 0:

                percentage = (
                    feature_importances[i]
                    / selected_importance
                ) * confidence

            else:

                percentage = (
                    confidence
                    / len(selected_indexes)
                )

            calculated_percentages.append(
                round(percentage, 1)
            )

        # -----------------------------------------
        # Make sure total = Model Confidence
        # -----------------------------------------

        current_total = round(
            sum(calculated_percentages),
            1
        )

        difference = round(
            confidence - current_total,
            1
        )

        # Add rounding difference to last factor
        last_index = len(calculated_percentages) - 1

        calculated_percentages[last_index] = round(
            calculated_percentages[last_index]
            + difference,
            1
        )

        # Store percentages
        for position, index in enumerate(selected_indexes):

            feature_percentages[
                feature_names[index]
            ] = calculated_percentages[position]


    # -----------------------------------------
    # Assessment Summary
    # -----------------------------------------

    summary = {

        "Threats": {
            "value": "Yes" if threats == 1 else "No",
            "percentage": feature_percentages["Threats"]
        },

        "Physical Harm": {
            "value": "Yes" if physical_harm == 1 else "No",
            "percentage": feature_percentages["Physical Harm"]
        },

        "Controlling Behavior": {
            "value":
                "Yes"
                if controlling_behavior == 1
                else "No",

            "percentage":
                feature_percentages["Controlling Behavior"]
        },

        "Verbal Abuse": {
            "value": "Yes" if verbal_abuse == 1 else "No",
            "percentage": feature_percentages["Verbal Abuse"]
        },

        "Isolation": {
            "value": "Yes" if isolation == 1 else "No",
            "percentage": feature_percentages["Isolation"]
        },

        "Financial Control": {
            "value":
                "Yes"
                if financial_control == 1
                else "No",

            "percentage":
                feature_percentages["Financial Control"]
        }
    }


    # -----------------------------------------
    # Save Assessment
    # -----------------------------------------

    save_assessment(
        threats,
        physical_harm,
        controlling_behavior,
        verbal_abuse,
        isolation,
        financial_control,
        risk
    )


    # -----------------------------------------
    # Temporary History
    # -----------------------------------------

    assessment_history.append({

        "risk": risk,

        "time": datetime.now().strftime(
            "%d-%m-%Y %I:%M %p"
        )

    })


    # -----------------------------------------
    # Result Page
    # -----------------------------------------

    return render_template(
        "result.html",

        risk=risk,

        summary=summary,

        confidence=confidence
    )


# -----------------------------------------
# Dashboard
# -----------------------------------------

@app.route("/dashboard")
def dashboard():

    # Read model accuracy
    with open("model/accuracy.txt", "r") as file:

        accuracy = float(file.read()) * 100


    # Connect database
    connection = sqlite3.connect(
        "risk_assessments.db"
    )

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()


    # Total assessments
    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM assessments
    """)

    total = cursor.fetchone()["total"]


    # Low Risk
    cursor.execute("""
        SELECT COUNT(*)
        FROM assessments
        WHERE risk_level = 'Low Risk'
    """)

    low = cursor.fetchone()[0]


    # Medium Risk
    cursor.execute("""
        SELECT COUNT(*)
        FROM assessments
        WHERE risk_level = 'Medium Risk'
    """)

    medium = cursor.fetchone()[0]


    # High Risk
    cursor.execute("""
        SELECT COUNT(*)
        FROM assessments
        WHERE risk_level = 'High Risk'
    """)

    high = cursor.fetchone()[0]


    # Assessment history
    cursor.execute("""
        SELECT *
        FROM assessments
        ORDER BY id DESC
    """)

    history = cursor.fetchall()

    connection.close()


    # Dashboard Page
    return render_template(
        "dashboard.html",

        total=total,

        low=low,

        medium=medium,

        high=high,

        history=history,

        accuracy=round(
            accuracy,
            1
        )
    )


# -----------------------------------------
# Run Flask Application
# -----------------------------------------

if __name__ == "__main__":
    app.run(debug=True)