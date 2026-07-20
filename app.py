import os
import joblib
import gradio as gr

# ==========================================================
# Load Trained Model
# ==========================================================
try:
    model = joblib.load("salary_prediction_model.pkl")
except Exception as e:
    print("Error loading model:", e)
    model = None

# ==========================================================
# Prediction Function
# ==========================================================
def predict_salary(years_experience):

    # Empty value check
    if years_experience is None or str(years_experience).strip() == "":
        return "❌ Please enter Years of Experience."

    # Numeric validation
    try:
        years_experience = float(years_experience)
    except:
        return "❌ Please enter a valid numeric value."

    # Negative validation
    if years_experience < 0:
        return "❌ Years of Experience cannot be negative."

    # Maximum validation
    if years_experience > 50:
        return "❌ Please enter experience between 0 and 50 years."

    # Model check
    if model is None:
        return "❌ Model not found."

    try:
        prediction = model.predict([[years_experience]])

        salary = prediction[0]

        return f"""
✅ Salary Prediction Successful

Years of Experience : {years_experience:.1f}

Predicted Salary

₹ {salary:,.2f}
"""

    except Exception as e:
        return f"Prediction Failed\n\n{e}"

# ==========================================================
# Description
# ==========================================================
DESCRIPTION = """
# 💰 Salary Prediction using Linear Regression

This application predicts the salary based on the employee's Years of Experience using a trained Linear Regression Machine Learning model.

---

## 👩‍💻 Developed By

**Manya**

---

## 🏫 College

Panipat Institute of Engineering & Technology (PIET)

---

## 🛠 Technologies Used

- Python
- Machine Learning
- Linear Regression
- Scikit-Learn
- Pandas
- NumPy
- Joblib
- Gradio

---

## Input

- Years of Experience

---

## Output

- Predicted Salary
"""

# ==========================================================
# Gradio Interface
# ==========================================================
interface = gr.Interface(
    fn=predict_salary,
    inputs=gr.Number(
        label="Years of Experience",
        value=2
    ),
    outputs=gr.Textbox(
        label="Predicted Salary",
        lines=8
    ),
    title="💰 Salary Prediction System",
    description=DESCRIPTION,
)

# ==========================================================
# Launch
# ==========================================================
if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
