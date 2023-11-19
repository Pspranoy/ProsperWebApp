#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pickle
import pandas as pd
import numpy as np


st.set_page_config(layout="wide")

# Load the trained models from pickle files
logistic_model = pickle.load(open('pipeline_class2.pkl', 'rb'))
linear_model = pickle.load(open('pipeline_reg2.pkl', 'rb'))

# Function to predict loan default using Logistic Regression
def predict_loan_default(input_data):
    prediction1 = logistic_model.predict(input_data)
    return prediction1

# Function to predict EMI, ELA, and PROI using Linear Regression
def predict_loan_metrics(input_data):
    prediction2 = linear_model.predict(input_data)
    prediction2 = prediction2.reshape([3, 1])
    m = np.array(prediction2)
    n =m.flatten()
    list(n)
    return (n)


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://plus.unsplash.com/premium_photo-1682109363124-26716c9db5b4?q=80&w=1932&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()
st.markdown("<h1 style='text-align: center; color: white;'>Loan Risk Predictor</h1>", unsafe_allow_html=True)

html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">PROSPER LOAN DATA </h2>
</div>
"""
        
st.markdown(html_temp, unsafe_allow_html=True)

st.title('CSV File Input')

# File upload functionality
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])


st.subheader("Fill in the Borrower's Information :")


# getting the input data from the user
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    EstimatedReturn = st.number_input("**EstimatedReturn**", min_value=0.0)
with col2:
    ListingCategory_numeric = st.number_input("**ListingCategory (numeric)**", min_value=0, max_value=20)
with col3:
    EmploymentStatusDuration = st.number_input("**Employment Status Duration**", min_value=0.0)

with col4:
    TotalInquiries = st.number_input("**Total Inquiries**", min_value=0)
with col5:
    CurrentDelinquencies = st.number_input("**Current Delinquencies**", min_value=0.0)
with col1:
    AmountDelinquent = st.number_input("**Amount Delinquent**", min_value=0.0)
with col2:
    DelinquenciesLast7Years = st.number_input("**Delinquencies in Last 7 Years**", min_value=0.0)
with col3:
    BankcardUtilization = st.number_input("**Bank Card Utilization**", min_value=0.0)
with col4:
    DebtToIncomeRatio = st.number_input("**Debt To Income Ratio**", min_value=0.00)
with col5:
    StatedMonthlyIncome = st.number_input("**Monthly Income**", min_value=0.0)
with col1:
    LoanCurrentDaysDelinquent = st.number_input("**Loan Current Days Delinquent**", min_value=0.0)
with col2:
    LoanOriginalAmount = st.number_input("**Applied Loan Amount**", min_value=0.0)
with col3:
    LP_CustomerPrincipalPayments = st.number_input("**Principal Amount**", min_value=0.0)
with col4:
    LP_GrossPrincipalLoss = st.number_input("**Gross Principal Loss**", min_value=0.0)

with col5:
    AvgCreditScore = st.number_input("**Average Credit Score**", min_value=0)
with col1:
    Tenure = st.number_input("**Loan Tenure**", min_value=0)

with col2:
    CreditGrade = st.selectbox('**Select Credit Grade**',['C', 'HR', 'AA', 'D', 'B', 'E', 'A', 'NC'])
with col3:
    BorrowerState = st.selectbox('**Select State Code**',['CO', 'GA', 'FL', 'MI', 'IL', 'NY', 'CA', 'MO', 'NE', 'KS',
   'VA', 'MN', 'MD', 'WI', 'OH', 'PA', 'AL', 'WA', 'NJ', 'TX', 'SC',
   'CT', 'KY', 'AZ', 'OK', 'OR', 'NC', 'MA', 'AR', 'TN', 'NM', 'ID',
   'DC', 'WV', 'NV', 'UT', 'MT', 'IN', 'NH', 'VT', 'LA', 'ME', 'AK',
   'HI', 'RI', 'WY', 'DE', 'IA', 'SD', 'MS', 'ND'])
with col4:
    Occupation = st.selectbox('**Select Occupation**',['Other', 'Waiter/Waitress', 'Professional', 'Skilled Labor',
   'Sales - Commission', 'Executive', 'Accountant/CPA',
   'Construction', 'Analyst', "Nurse's Aide", 'Fireman', 'Realtor',
   'Clerical', 'Laborer', 'Food Service Management', 'Truck Driver',
   'Administrative Assistant', 'Police Officer/Correction Officer',
   'Nurse (RN)', 'Social Worker', 'Computer Programmer',
   'Military Officer', 'Sales - Retail', 'Military Enlisted',
   'Food Service', 'Tradesman - Mechanic', 'Postal Service',
   'Teacher', 'Pharmacist', 'Retail Management',
   'Engineer - Mechanical', 'Dentist', 'Architect', 'Landscaping',
   'Nurse (LPN)', 'Tradesman - Carpenter', 'Medical Technician',
   'Tradesman - Plumber', 'Tradesman - Electrician', 'Bus Driver',
   'Engineer - Chemical', 'Student - College Senior', 'Principal',
   'Attorney', 'Scientist', 'Doctor', 'Pilot - Private/Commercial',
   'Engineer - Electrical', 'Homemaker',
   'Student - College Graduate Student', 'Civil Service',
   'Student - Technical School', 'Psychologist', 'Biologist',
   'Religious', 'Professor', 'Chemist', 'Student - College Sophomore',
   'Clergy', 'Investor', 'Student - College Junior',
   'Flight Attendant', 'Car Dealer', "Teacher's Aide",
   'Student - Community College', 'Student - College Freshman',
   'Judge'])
with col5:
    EmploymentStatus = st.selectbox('**Select Employment Status**',['Self-employed', 'Not available', 'Full-time', 'Other', 'Employed', 'Not employed', 'Part-time', 'Retired'])
with col1:
    IsBorrowerHomeowner = st.radio('**Is Borrower a Homeowner?**', ['True','False'],horizontal=True)
with col2:
    IncomeVerifiable = st.radio('**Is Income Verifiable?**', ['True','False'],horizontal=True)
with col3:
    BorrowerAPR = st.number_input("**Borrower APR**", min_value=0.0)
with col4:
    EstimatedEffectiveYield = st.number_input("**Estimated Effective Yield**", min_value=0.0)
with col5:
    EstimatedLoss = st.number_input("**Estimated Loss**", min_value=0.0)
#with col1:
#    OpenCreditLines = st.number_input("**OpenCreditLines**", min_value=0.0)

    
result1 = ""
result2 = ""

default_html="""  
  <div style="background-color:#F4D03F;padding:10px >
   <h2 style="color:white;text-align:center;"> Your Loan Status is Default</h2>
   </div>
"""
notdefault_html="""  
  <div style="background-color:#F08080;padding:10px >
   <h2 style="color:black ;text-align:center;"> Your Loan Status is Not Default</h2>
   </div>
"""

input=[CreditGrade, BorrowerAPR, EstimatedEffectiveYield,
   EstimatedLoss, EstimatedReturn, ListingCategory_numeric,
   BorrowerState, Occupation, EmploymentStatus,
   EmploymentStatusDuration, IsBorrowerHomeowner, TotalInquiries,
   CurrentDelinquencies,
   AmountDelinquent, DelinquenciesLast7Years, BankcardUtilization,
   DebtToIncomeRatio, IncomeVerifiable, StatedMonthlyIncome,
   LoanCurrentDaysDelinquent, LoanOriginalAmount,
   LP_CustomerPrincipalPayments, LP_GrossPrincipalLoss,
   AvgCreditScore, Tenure]



# changing the input_data to numpy array
input1 =np.asarray(input)

input_data = input1.reshape(1,-1)

if uploaded_file is not None:
    data_entered = pd.read_csv(uploaded_file)
else:
    data_entered = input_data
    
st.write("You entered:", data_entered)
    
#col1, col2= st.columns([1,1])



with col1:
    if st.button("Loan Status"):
        result1 = predict_loan_default(data_entered)
        if result1 == 1:
            st.success('**Your Loan Status is Not Default.**')
            st.markdown("![Loan Will Not Default!](https://media.giphy.com/media/TIcMOKgkoCVtTcpIFb/giphy.gif)")
            st.balloons()
            st.session_state.clicked = True
            
            
        else:
            st.error('**Your loan Status is Default**', icon="🚨")
            st.markdown("![Loan Will Default!](https://media.giphy.com/media/gKAsvFhqOmVI8nF4ro/giphy.gif)")
            st.toast('**Loan will Default!**')
            st.session_state.clicked = False
            
with col2:
            if st.button("Prediction"):
                result2 = predict_loan_metrics(data_entered)
                a=float(result2[0])
                a=round(a,2)
                b=float(result2[1])
                b=round(b,2)
                c=float(result2[2])
                c=round(c,2)
                st.info("**Preferred Equated Monthly Installment (EMI) is : $** {}".format(a))
                st.info("**Eligible Loan Amount (ELA) is : $** {}".format(b))
                st.info("**Preferred Return On Investment (PROI) is :** {} % ".format(c))
                
