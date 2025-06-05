import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Run the app with: streamlit run "F:/Complete ML/All_Projects/MLProject3/app.py"

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# 👉👉👉 ADD THIS LINE BELOW 👇👇👇
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Multiple Disease Prediction System</h1>", unsafe_allow_html=True)


st.markdown("""
<style>
    /* Sidebar width */
    section[data-testid="stSidebar"] {
        width: 350px !important;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        background-color: #f0f0f0;
        color: black;
        border-radius: 8px;
        caret-color: #222 !important;
    }

    /* Placeholder styling */
    .stTextInput > div > div > input::placeholder {
        color: #888 !important;
        opacity: 1 !important;
    }

    /* Help text styling */
    .css-1b0udgb {
        color: black !important;
        font-size: 0.85rem;
    }

    /* Stylish buttons */
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }

    /* REMOVE TOP MARGIN */
    .block-container {
        padding-top: 1rem !important;
        margin-top: 0rem !important;
    }
    header[data-testid="stHeader"] {
        height: 0px !important;
        min-height: 0px !important;
        visibility: hidden;
        display: none;
    }
            
    .success-box {
        background-color: #06290e;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #28a745;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load models
# diabetes_model = pickle.load(open(r'F:\Complete ML\All_Projects\MLProject3\model\diabetes_model.sav', 'rb'))
# heart_disease_model = pickle.load(open(r'F:\Complete ML\All_Projects\MLProject3\model\heart_disease_model.sav', 'rb'))
# parkinsons_model = pickle.load(open(r'F:\Complete ML\All_Projects\MLProject3\model\parkinsons_model.sav', 'rb'))
# breast_cancer_model = pickle.load(open(r'F:\Complete ML\All_Projects\MLProject3\model\breast_cancer.sav', 'rb'))

heart_disease_model = pickle.load(open('model/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('model/parkinsons_model.sav', 'rb'))
breast_cancer_model = pickle.load(open('model/breast_cancer.sav', 'rb'))
diabetes_model = pickle.load(open('model/diabetes_model.sav', 'rb'))

with st.sidebar:
    selected = option_menu(
        'Select disease',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Breast Cancer Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person', 'file-medical'],
        default_index=0
    )

    # Medical Disclaimer
    st.header("⚠️ Medical Disclaimer")
    st.markdown("""
    <div class="warning-box">
    <p><strong>Important:</strong> This tool is for research and educational purposes only. 
    It should not be used for actual medical diagnosis. Always consult with qualified 
    healthcare professionals for medical decisions.</p>
    </div>
    <style>
    .warning-box {
        background-color: #2e260c;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #ffc107;
    }
    </style>
    """, unsafe_allow_html=True)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('🩺 Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies', placeholder='e.g. 2')
    with col2:
        Glucose = st.text_input('Glucose Level', placeholder='e.g. 120')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value', placeholder='e.g. 80')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value', placeholder='e.g. 20')
    with col2:
        Insulin = st.text_input('Insulin Level', placeholder='e.g. 85')
    with col3:
        BMI = st.text_input('BMI value', placeholder='e.g. 26.5')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', placeholder='e.g. 0.5')
    with col2:
        Age = st.text_input('Age of the Person', placeholder='e.g. 45')

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                    BMI, DiabetesPedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        diab_prediction = diabetes_model.predict([user_input])
        if diab_prediction[0] == 1:
            diab_diagnosis = '⚠️ The person is <b>diabetic</b>'
        else:
            diab_diagnosis = '✅ The person is <b>not diabetic</b>'

    if diab_diagnosis:
        st.markdown(f'<div class="success-box">{diab_diagnosis}</div>', unsafe_allow_html=True)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age', placeholder='e.g. 54')
    with col2:
        sex = st.text_input('Sex (1=Male, 0=Female)', placeholder='e.g. 1')
    with col3:
        cp = st.text_input('Chest Pain Type (0–3)', placeholder='e.g. 2')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure', placeholder='e.g. 130')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl', placeholder='e.g. 250')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)', placeholder='e.g. 0')
    with col1:
        restecg = st.text_input('Resting ECG Result (0–2)', placeholder='e.g. 1')
    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved', placeholder='e.g. 160')
    with col3:
        exang = st.text_input('Exercise Induced Angina (1=Yes, 0=No)', placeholder='e.g. 0')
    with col1:
        oldpeak = st.text_input('ST Depression Induced by Exercise', placeholder='e.g. 1.2')
    with col2:
        slope = st.text_input('Slope of the Peak ST Segment (0–2)', placeholder='e.g. 1')
    with col3:
        ca = st.text_input('Major Vessels Colored by Flourosopy (0–3)', placeholder='e.g. 0')
    with col1:
        thal = st.text_input('Thal (0=normal, 1=fixed defect, 2=reversible defect)', placeholder='e.g. 2')

    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg,
                    thalach, exang, oldpeak, slope, ca, thal]
        user_input = [float(x) for x in user_input]
        heart_prediction = heart_disease_model.predict([user_input])
        if heart_prediction[0] == 1:
            heart_diagnosis = '❤️ The person <b>has heart disease</b>'
        else:
            heart_diagnosis = '💚 The person <b>does not have any heart disease</b>'

    if heart_diagnosis:
        st.markdown(f'<div class="success-box">{heart_diagnosis}</div>', unsafe_allow_html=True)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.title("🧠 Parkinson's Disease Prediction using ML")

    inputs = {}
    parkinsons_features = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
        'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
        'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
        'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]

    cols = st.columns(5)
    for i, feature in enumerate(parkinsons_features):
        with cols[i % 5]:
            inputs[feature] = st.text_input(feature, placeholder='e.g. 119.992')

    parkinsons_diagnosis = ''
    if st.button("Parkinson's Test Result"):
        user_input = [float(inputs[f]) for f in parkinsons_features]
        parkinsons_prediction = parkinsons_model.predict([user_input])
        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "🧠 The person <b>has Parkinson's disease</b>"
        else:
            parkinsons_diagnosis = "🧠 The person <b>does not have Parkinson's disease</b>"

    if parkinsons_diagnosis:
        st.markdown(f'<div class="success-box">{parkinsons_diagnosis}</div>', unsafe_allow_html=True)

# Breast Cancer Prediction Page
if selected == 'Breast Cancer Prediction':
    st.title('👩‍⚕️ Breast Cancer Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        radius_mean = st.text_input('Radius Mean', placeholder='e.g. 17.99')
    with col2:
        texture_mean = st.text_input('Texture Mean', placeholder='e.g. 10.38')
    with col3:
        perimeter_mean = st.text_input('Perimeter Mean', placeholder='e.g. 122.8')
    with col1:
        area_mean = st.text_input('Area Mean', placeholder='e.g. 1001')
    with col2:
        smoothness_mean = st.text_input('Smoothness Mean', placeholder='e.g. 0.1184')
    with col3:
        compactness_mean = st.text_input('Compactness Mean', placeholder='e.g. 0.2776')
    with col1:
        concavity_mean = st.text_input('Concavity Mean', placeholder='e.g. 0.3001')
    with col2:
        concave_points_mean = st.text_input('Concave Points Mean', placeholder='e.g. 0.1471')
    with col3:
        symmetry_mean = st.text_input('Symmetry Mean', placeholder='e.g. 0.2419')
    with col1:
        fractal_dimension_mean = st.text_input('Fractal Dimension Mean', placeholder='e.g. 0.07871')

    breast_cancer_diagnosis = ''
    if st.button('Breast Cancer Test Result'):
        user_input = [radius_mean, texture_mean, perimeter_mean, area_mean,
                    smoothness_mean, compactness_mean, concavity_mean,
                    concave_points_mean, symmetry_mean, fractal_dimension_mean]
        user_input = [float(x) for x in user_input]
        breast_cancer_prediction = breast_cancer_model.predict([user_input])
        if breast_cancer_prediction[0] == 1:
            breast_cancer_diagnosis = '🩺 The person <b>has Breast Cancer</b>'
        else:
            breast_cancer_diagnosis = '🩺 The person <b>does not have Breast Cancer</b>'

    if breast_cancer_diagnosis:
        st.markdown(f'<div class="success-box">{breast_cancer_diagnosis}</div>', unsafe_allow_html=True)
