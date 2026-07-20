import os
import joblib
import gradio as gr

# ==========================================================
# Load Trained Model
# ==========================================================

try:
    model = joblib.load("salary_prediction_model.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# ==========================================================
# Prediction Function
# ==========================================================

def predict_salary(years_experience):

    if years_experience is None:
        return "❌ Please enter Years of Experience."

    try:
        years_experience = float(years_experience)
    except ValueError:
        return "❌ Please enter a valid numeric value."

    if years_experience < 0:
        return "❌ Years of Experience cannot be negative."

    if years_experience > 50:
        return "❌ Please enter a value between 0 and 50."

    if model is None:
        return "❌ Model file not found."

    try:
        prediction = model.predict([[years_experience]])

        salary = prediction[0]

        return f"""
✅ Salary Prediction Completed

Years of Experience : {years_experience:.1f} Years

Predicted Salary

₹ {salary:,.2f}
"""

    except Exception as e:
        return f"Prediction Error\n\n{e}"

# ==========================================================
# Description
# ==========================================================

DESCRIPTION = """
# 💰 Salary Prediction System

This application predicts the **Salary** of an employee based on the **Years of Experience** using a trained **Linear Regression Machine Learning Model**.

---

## 👩‍💻 Developed By

**Manya Singla**

---

## 🏫 College

**Panipat Institute of Engineering & Technology (PIET), Panipat**

---

## 🎓 Project

**Salary Prediction using Machine Learning**

---

## 🔗 GitHub Repository

https://github.com/Manya2507/Salary-Prediction-Model

---

## 🛠 Technologies Used

• Python

• Machine Learning

• Linear Regression

• Scikit-learn

• Pandas

• NumPy

• Joblib

• Gradio

• Git & GitHub

---

## 📋 Instructions

1. Enter Years of Experience.

2. Click Submit.

3. The model predicts the estimated salary.

---

## 📌 Input

• Years of Experience

---

## 📈 Output

• Predicted Salary
"""

# ==========================================================
# Interface
# ==========================================================

demo = gr.Interface(
    fn=predict_salary,
    inputs=gr.Number(
        label="Years of Experience",
        precision=1
    ),
    outputs=gr.Textbox(
        label="Prediction Result",
        lines=8
    ),
    title="💰 Salary Prediction using Linear Regression",
    description=DESCRIPTION
)

# ==========================================================
# Launch
# ==========================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
