import os
import joblib
import pandas as pd
import gradio as gr

# =====================================
# Load Trained Model
# =====================================
model = joblib.load("salary_prediction_model.pkl")

# =====================================
# Categories used while training
# =====================================

education_categories = [
    "Bachelor's",
    "Bachelor's Degree",
    "High School",
    "Master's",
    "Master's Degree",
    "PhD",
    "phD",
    "others"
]

job_categories = [
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
    "others"
]

# =====================================
# Prediction Function
# =====================================

def predict_salary(age, experience, education, job):

    data = {
        "Age": age,
        "Years of Experience": experience,

        "Education Level_Bachelor's": 0,
        "Education Level_Bachelor's Degree": 0,
        "Education Level_High School": 0,
        "Education Level_Master's": 0,
        "Education Level_Master's Degree": 0,
        "Education Level_PhD": 0,
        "Education Level_others": 0,
        "Education Level_phD": 0,

        "Job Title_Back end Developer": 0,
        "Job Title_Data Analyst": 0,
        "Job Title_Data Scientist": 0,
        "Job Title_Full Stack Engineer": 0,
        "Job Title_Marketing Manager": 0,
        "Job Title_Product Manager": 0,
        "Job Title_Senior Project Engineer": 0,
        "Job Title_Senior Software Engineer": 0,
        "Job Title_Software Engineer": 0,
        "Job Title_Software Engineer Manager": 0,
        "Job Title_others": 0
    }

    # Education Encoding
    if education == "Bachelor's":
        data["Education Level_Bachelor's"] = 1
    elif education == "Bachelor's Degree":
        data["Education Level_Bachelor's Degree"] = 1
    elif education == "High School":
        data["Education Level_High School"] = 1
    elif education == "Master's":
        data["Education Level_Master's"] = 1
    elif education == "Master's Degree":
        data["Education Level_Master's Degree"] = 1
    elif education == "PhD":
        data["Education Level_PhD"] = 1
    elif education == "phD":
        data["Education Level_phD"] = 1
    else:
        data["Education Level_others"] = 1

    # Job Encoding
    if job == "Back end Developer":
        data["Job Title_Back end Developer"] = 1
    elif job == "Data Analyst":
        data["Job Title_Data Analyst"] = 1
    elif job == "Data Scientist":
        data["Job Title_Data Scientist"] = 1
    elif job == "Full Stack Engineer":
        data["Job Title_Full Stack Engineer"] = 1
    elif job == "Marketing Manager":
        data["Job Title_Marketing Manager"] = 1
    elif job == "Product Manager":
        data["Job Title_Product Manager"] = 1
    elif job == "Senior Project Engineer":
        data["Job Title_Senior Project Engineer"] = 1
    elif job == "Senior Software Engineer":
        data["Job Title_Senior Software Engineer"] = 1
    elif job == "Software Engineer":
        data["Job Title_Software Engineer"] = 1
    elif job == "Software Engineer Manager":
        data["Job Title_Software Engineer Manager"] = 1
    else:
        data["Job Title_others"] = 1

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    return f"Predicted Salary: ₹ {prediction:,.0f}"

# =====================================
# CSS
# =====================================

css = """
.gradio-container{
background-image:url("https://images.unsplash.com/photo-1520607162513-77705c0f0d4a");
background-size:cover;
background-position:center;
background-attachment:fixed;
}

.glass{
background:rgba(255,255,255,0.92);
padding:20px;
border-radius:15px;
}

.gr-button{
background:#2563eb !important;
color:white !important;
}
"""

# =====================================
# Interface
# =====================================

with gr.Blocks(css=css,title="Salary Prediction System") as demo:

    with gr.Column(elem_classes="glass"):

        gr.Markdown(
        """
# 💼 Salary Prediction System

Predict employee salary using Machine Learning.
        """
        )

        with gr.Row():

            with gr.Column():

                age = gr.Number(label="Age")

                experience = gr.Number(label="Years of Experience")

                education = gr.Dropdown(
                    choices=education_categories,
                    value="Bachelor's",
                    label="Education Level"
                )

                job = gr.Dropdown(
                    choices=job_categories,
                    value="Software Engineer",
                    label="Job Title"
                )

                predict = gr.Button("Predict Salary")

                output = gr.Textbox(label="Prediction")

            with gr.Column():

                gr.Markdown("""
## 👩‍💻 Developer

**Name:** Manya Singla

**College:** Panipat Institute of Engineering and Technology

**Project:** AI Based Salary Prediction System

**Machine Learning Model:** Linear Regression

### Tools Used

- Python
- Gradio
- Scikit-Learn
- Pandas
- Joblib
                """)

        predict.click(
            predict_salary,
            inputs=[age, experience, education, job],
            outputs=output
        )

# =====================================
# Launch
# =====================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT",7860))
    )
