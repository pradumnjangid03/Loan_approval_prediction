from flask import Flask, render_template, request
import pandas as pd
from joblib import load
from model_train import MultiColumnLabelEncoder

app = Flask(__name__)
model = load("loan_pipeline_web.joblib")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    form_data = {
        "Gender": "",
        "Married": "",
        "Dependents": "",
        "Education": "",
        "Self_Employed": "",
        "ApplicantIncome": "",
        "CoapplicantIncome": "",
        "LoanAmount": "",
        "Loan_Amount_Term": "",
        "Credit_History": "",
        "Property_Area": ""
    }

    if request.method == "POST":
        form_data = {
            "Gender": request.form["Gender"],
            "Married": request.form["Married"],
            "Dependents": request.form["Dependents"],
            "Education": request.form["Education"],
            "Self_Employed": request.form["Self_Employed"],
            "ApplicantIncome": request.form["ApplicantIncome"],
            "CoapplicantIncome": request.form["CoapplicantIncome"],
            "LoanAmount": request.form["LoanAmount"],
            "Loan_Amount_Term": request.form["Loan_Amount_Term"],
            "Credit_History": request.form["Credit_History"],
            "Property_Area": request.form["Property_Area"]
        }

        input_df = pd.DataFrame([{
            "Gender": form_data["Gender"],
            "Married": form_data["Married"],
            "Dependents": form_data["Dependents"],
            "Education": form_data["Education"],
            "Self_Employed": form_data["Self_Employed"],
            "ApplicantIncome": float(form_data["ApplicantIncome"]),
            "CoapplicantIncome": float(form_data["CoapplicantIncome"]),
            "LoanAmount": float(form_data["LoanAmount"]),
            "Loan_Amount_Term": float(form_data["Loan_Amount_Term"]),
            "Credit_History": float(form_data["Credit_History"]),
            "Property_Area": form_data["Property_Area"]
        }])

        pred = model.predict(input_df)[0]
        prediction = "✅ Loan Approved" if pred == 1 else "❌ Loan Rejected"

    return render_template("form.html", prediction=prediction, form_data=form_data)

if __name__ == "__main__":
    app.run(debug=True)
