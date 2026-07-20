import os
import joblib
import gradio as gr

# ==========================================================
# Load Trained Model
# ==========================================================

try:
    model = joblib.load("salary_prediction_model.pkl")
except Exception as e:
    print(e)
    model = None


# ==========================================================
# Prediction Function
# ==========================================================

def predict_salary(
    age,
    experience,
    education,
    job_title,
):

    if model is None:
        return "❌ Model file not found."

    try:
        age = int(age)
        experience = float(experience)
    except:
        return "❌ Please enter valid numeric values."

    if age < 18 or age > 70:
        return "❌ Age should be between 18 and 70."

    if experience < 0 or experience > 50:
        return "❌ Years of Experience should be between 0 and 50."

    # -----------------------------------------------------
    # Feature Vector (22 Features)
    # -----------------------------------------------------

    features = [0] * 22

    # Numerical Features
    features[0] = age
    features[1] = experience

    # ---------------- Education ----------------

    education_map = {
        "Bachelor's": 2,
        "Bachelor's Degree": 3,
        "High School": 4,
        "Master's": 5,
        "Master's Degree": 6,
        "PhD": 7,
        "others": 8,
        "phD": 9,
    }

    if education in education_map:
        features[education_map[education]] = 1
    else:
        features[8] = 1

    # ---------------- Job Title ----------------

    job_map = {
        "Back end Developer": 10,
        "Data Analyst": 11,
        "Data Scientist": 12,
        "Full Stack Engineer": 13,
        "Marketing Manager": 14,
        "Product Manager": 15,
        "Senior Project Engineer": 16,
        "Senior Software Engineer": 17,
        "Software Engineer": 18,
        "Software Engineer Manager": 19,
        "Others": 20,
    }

    if job_title in job_map:
        features[job_map[job_title]] = 1
    else:
        features[20] = 1

    try:
        prediction = model.predict([features])[0]

        return f"""
## 💰 Salary Prediction

### Predicted Salary

₹ {prediction:,.2f} per year
"""

    except Exception as e:
        return f"Prediction Error\n\n{e}"


# ==========================================================
# Description
# ==========================================================

DESCRIPTION = """
# 💰 Salary Prediction System

Predict employee salary using Machine Learning.

---

## 👩‍💻 Developed By

**Manya**

---

## 🏫 College

Panipat Institute of Engineering & Technology (PIET)

---

## Technologies

- Python
- Machine Learning
- Scikit-Learn
- Gradio
- Joblib
- Pandas
- NumPy

---
## 🔗 GitHub Repository
https://github.com/Manya2507/Salary-Prediction-Model
---

## Required Inputs

- Age
- Years of Experience
- Education Level
- Job Title
"""

# ==========================================================
# Interface
# ==========================================================

demo = gr.Interface(
    fn=predict_salary,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Years of Experience"),
        gr.Dropdown(
            choices=[
                "Bachelor's",
                "Bachelor's Degree",
                "High School",
                "Master's",
                "Master's Degree",
                "PhD",
                "phD",
                "others",
            ],
            label="Education Level",
        ),
        gr.Dropdown(
            choices=[
                "Back end Developer",
                "Data Analyst",
                "Data Scientist",
                "Full Stack Engineer",
                "Marketing Manager",
                "Product Manager",
                "Senior Project Engineer",
                "Senior Software Engineer",
                "Software Engineer",
                "Software Engineer Manager",
                "Others",
            ],
            label="Job Title",
        ),
    ],
    outputs=gr.Markdown(),
    title="💰 Salary Prediction System",
    description=DESCRIPTION,
)

# ==========================================================
# Launch
# ==========================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
