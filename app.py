import streamlit as st
import pandas as pd
import joblib

# Gradient animated background with CSS
st.markdown("""
    <style>
        @keyframes gradientBackground {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        .stApp {
            background: linear-gradient(-45deg, #e66465, #9198e5, #89f7fe, #66a6ff);
            background-size: 600% 600%;
            animation: gradientBackground 15s ease infinite;
        }

        .main-box {
            background-color: rgba(255, 255, 255, 0.92);
            padding: 2rem;
            border-radius: 15px;
            max-width: 900px;
            margin: auto;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }

        h1, h3 {
            text-align: center;
            color: #2c3e50;
        } section[data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.9);  /* Semi-transparent white */
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);  /* Soft shadow */
        }

        section[data-testid="stSidebar"] label {
            color: #333333 !important;  /* Dark text */
            font-size: 16px;
            font-weight: 600;
            font-family: 'Poppins', sans-serif;
        }


        [data-testid="stSidebar"] h1 {
            color: #333333;
            font-family: 'Poppins', sans-serif;
            font-weight: bold;
        }

         /* Predict button */
        div.stButton > button {
        background-color: #6c757d;
        color: white;
        padding: 0.75rem 2rem;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transition: 0.3s ease;
 }

div.stButton > button:hover {
    background-color: #00997a;
    transform: scale(1.05);
}

     </style>
""", unsafe_allow_html=True)

# Load the trained model
model = joblib.load("best_model.pkl")
label_encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Employee Salary Prediction App", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Prediction App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on the input features entered by user.")

st.sidebar.markdown("<h1 class='sidebar-title'>Enter Employee Details</h1>", unsafe_allow_html=True)

# Sidebar inputs (these must match your training feature columns)

# ✨ Replace these fields with your dataset's actual input columns
age = st.sidebar.number_input("Age", min_value=17, max_value=70, value=25)
education = st.sidebar.selectbox("Education Level", [
    "Bachelors", "Masters", "Prof-school", "HS-grad", "Some-college", "Assoc-acdm", "Assoc-voc", "Doctorate"
])
occupation = st.sidebar.selectbox("Job Role", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)

native_country_list = ["United-States", "India", "Mexico", "Philippines", "Germany",
    "Canada", "England", "China", "Japan", "France", "Italy", 
    "Vietnam", "South", "Puerto-Rico", "Jamaica", 
    "El-Salvador", "Cuba", "Iran", "Ireland", "Poland", "Thailand",
    "Hong", "Cambodia", "Scotland", "Columbia", "Guatemala", 
    "Yugoslavia", "Dominican-Republic", "Portugal", "Nicaragua", 
    "Greece", "Ecuador", "Peru", "Outlying-US(Guam-USVI-etc)", 
    "Haiti", "Taiwan", "Not-Listed"]

native_country = st.sidebar.selectbox("Native Country", native_country_list)

gender = st.sidebar.radio("Gender", ["Male", "Female"])

marital_status = st.sidebar.selectbox("Marital Status", [
    "Married-civ-spouse","Never-married", "Divorced","Separated","Widowed","Married-spouse-absent","Married-AF-spouse" 
])
race = st.sidebar.selectbox("Race",["White","Black","Asian-Pac-Islander","Amer-Indian-Eskimo","Other"])

relationship = st.sidebar.selectbox("Relationship",["Husband","Not-in-family","Own-child","Unmarried","Wife", "Other-relative"])

workclass = st.sidebar.selectbox("Work_Class",["Private","Self-emp-not-inc","Local-gov","Others","State-gov", "Self-emp-inc", "Federal-gov"])


#features whose values can't be taken as input from the user- for user friendliness
# 1. fnlwgt (mean value)
fnlwgt = 189664.13459727284 # Replace with actual mean from your dataset

# 2. capital-gain & capital-loss (defaults)
capital_gain = 0  #0 based on maximum training data
capital_loss = 0

# 3. education-num (mapped from education)
education_map = {
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-acdm": 11,
    "Assoc-voc": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Doctorate": 16,
    "Prof-school" : 15
}
education_num = education_map[education]

# Build input DataFrame (⚠️ must match preprocessing of your training data)
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt' : [fnlwgt],
    'education': [education],
    'educational-num': [education_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship' : [relationship],
    'race' : [race],
    'gender': [gender],
    'capital-gain' : [capital_gain],
    'capital-loss' : [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country],

})
raw_input_df = input_df.copy()

# Apply label encoding to all categorical columns
for col in input_df.columns:
    if col in label_encoders:
        try:
            input_df[col] = label_encoders[col].transform(input_df[col])
        except ValueError as e:
            st.error(f"ValueError in column '{col}': {e}")
            st.stop()

st.write("🔍 User Input Details:", raw_input_df)


# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.markdown(f"""
        <div style='
            background-color: rgba(255,255,255,0.95);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            color: #00b894;
            margin-top: 20px;
        '>
        💰 Predicted Salary Category: {prediction[0]}
        </div>
    """, unsafe_allow_html=True)  
