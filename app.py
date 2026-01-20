import gradio as gr
import pandas as pd
import pickle


with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)


def predict_loan(Gender, Married, Dependents, Education, Self_Employed,
                 ApplicantIncome, CoapplicantIncome, LoanAmount,
                 Loan_Amount_Term, Credit_History, Property_Area):

    input_df = pd.DataFrame([[
        Gender, Married, Dependents, Education, Self_Employed,
        ApplicantIncome, CoapplicantIncome, LoanAmount,
        Loan_Amount_Term, Credit_History, Property_Area,
        ApplicantIncome + CoapplicantIncome  # TotalIncome
    ]],
        columns=[
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'TotalIncome'
    ])

    pred = model.predict(input_df)[0]

    return "Approved " if pred == 1 else "Not Approved"


inputs = [
    gr.Dropdown(["Male", "Female"], label="Gender"),
    gr.Dropdown(["Yes", "No"], label="Married"),
    gr.Dropdown(["0", "1", "2", "3+"], label="Dependents"),
    gr.Dropdown(["Graduate", "Not Graduate"], label="Education"),
    gr.Dropdown(["Yes", "No"], label="Self Employed"),
    gr.Number(label="Applicant Income"),
    gr.Number(label="Coapplicant Income"),
    gr.Number(label="Loan Amount"),
    gr.Number(label="Loan Amount Term (months)"),
    gr.Dropdown([0, 1], label="Credit History"),
    gr.Dropdown(["Urban", "Rural", "Semiurban"], label="Property Area")
]

app = gr.Interface(
    fn=predict_loan,
    inputs=inputs,
    outputs="text",
    title="Loan Approval Prediction"
)

app.launch(share=True)
