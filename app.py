import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define model input dimension
MODEL_INPUT_DIM = 11

# Define the model architecture to match training
class LoanModel(nn.Module):
    def __init__(self, model_input_dim):
        super(LoanModel, self).__init__()
        self.fc1 = nn.Linear(model_input_dim, 32)
        self.dropout1 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Load the model
model = LoanModel(model_input_dim=MODEL_INPUT_DIM)
model.load_state_dict(torch.load("loan_approval_model.pth", map_location=torch.device("cpu")))
model.eval()

# Prepare encoders
encoders = {}
categorical_cols = ['gender', 'married', 'dependents', 'education', 'self_employed', 'property_area']
for col in categorical_cols:
    encoders[col] = LabelEncoder()
    if col == 'gender':
        encoders[col].fit(['Male', 'Female'])
    elif col == 'married':
        encoders[col].fit(['No', 'Yes'])
    elif col == 'dependents':
        encoders[col].fit(['0', '1', '2', '3+'])
    elif col == 'education':
        encoders[col].fit(['Graduate', 'Not Graduate'])
    elif col == 'self_employed':
        encoders[col].fit(['No', 'Yes'])
    elif col == 'property_area':
        encoders[col].fit(['Urban', 'Semiurban', 'Rural'])

# Dummy scaler initialization
scaler = StandardScaler()

# Streamlit UI
st.title('Loan Eligibility Calculator 🏦')

Name = st.text_input('Enter Your Name')
Gender = st.selectbox('Gender', ('Male', 'Female'))
Married = st.selectbox('Married', ('No', 'Yes'))
Dependents = st.selectbox('Number Of Dependents', ('0', '1', '2', '3 or More Dependents'))
Education = st.selectbox('Education status', ('Graduate', 'Not Graduate'))
Self_Employed = st.selectbox('Self Employed', ('No', 'Yes'))
ApplicantIncome = st.number_input('Applicant Income', min_value=10000)
CoapplicantIncome = st.number_input('Coapplicant Income', 0)
LoanAmount = st.number_input('Loan Amount', min_value=100000)
Loan_Amount_Term = st.select_slider('Loan Amount Term', ['1 YEAR', '3 YEARS', '5 YEARS', '7 YEARS',
                                                         '10 YEARS', '15 YEARS', '20 YEARS', '25 YEARS',
                                                         '30 YEARS', '40 YEARS'])
credit_score = st.number_input('Credit Score (300 - 850)', 300, 850)
Property_Area = st.selectbox('Area of Property', ('Urban', 'Rural', 'Semiurban'))

dependents_map = {'0': '0', '1': '1', '2': '2', '3 or More Dependents': '3+'}
term_map = {'1 YEAR': 12, '3 YEARS': 36, '5 YEARS': 60, '7 YEARS': 84, '10 YEARS': 120,
            '15 YEARS': 180, '20 YEARS': 240, '25 YEARS': 300, '30 YEARS': 360, '40 YEARS': 480}
property_map = {'Urban': 'Urban', 'Semiurban': 'Semiurban', 'Rural': 'Rural'}

result_text = ""

if st.button('Predict'):
    input_data = {
        'gender': Gender,
        'married': Married,
        'dependents': dependents_map[Dependents],
        'education': Education,
        'self_employed': Self_Employed,
        'applicantincome': ApplicantIncome,
        'coapplicantincome': CoapplicantIncome,
        'loanamount': LoanAmount,
        'loan_amount_term': term_map[Loan_Amount_Term],
        'credit_score': credit_score,
        'property_area': property_map[Property_Area]
    }

    input_df = pd.DataFrame([input_data])

    for col in categorical_cols:
        input_df[col] = encoders[col].transform(input_df[col])

    numerical_cols = ['applicantincome', 'coapplicantincome', 'loanamount', 'loan_amount_term', 'credit_score']
    input_scaled = input_df[numerical_cols + categorical_cols].values

    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        nn_pred = model(input_tensor)
        nn_result = "✅ Loan Approved!" if nn_pred.item() > 0.5 else "❌ Loan Not Approved!"

    num_dependents = int(dependents_map[Dependents].replace('3+', '3'))
    loan_term_months = term_map[Loan_Amount_Term]
    total_monthly_income = ApplicantIncome + CoapplicantIncome
    total_income_during_term = total_monthly_income * loan_term_months

    if credit_score >= 600 and num_dependents < 3 and total_income_during_term >= LoanAmount:
        rule_result = " Loan Approved!✅"
    else:
        rule_result = " Loan Not Approved! ❌ "

    result_text = f"{rule_result}"

if result_text:
    st.markdown("---")
    st.subheader("Prediction Result")
    for line in result_text.split('\n'):
        if "✅" in line:
            st.success(line)
        else:
            st.error(line)

# Style the button
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #0099ff;
        color:#ffffff;
    }
    div.stButton > button:hover {
        background-color: #00ff00;
        color:#ff0000;
    }
    </style>
""", unsafe_allow_html=True)
