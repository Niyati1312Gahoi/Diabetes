import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"./diabetes.csv")

# st.markdown("<h1 style='text-align: center; color: #FF9F40;'>Diabetes Prediction App</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center;'>This app predicts whether a patient is diabetic based on their health data.</p>", unsafe_allow_html=True)
# st.markdown("---")
# Title with a gradient background and animation
st.markdown(
    """
    <style>
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .title {
            text-align: center;
            color: white;
            font-size: 2.5em;
            padding: 10px;
            background: linear-gradient(45deg, #FF9F40, #F77B91, #FFC048);
            background-size: 200% 200%;
            animation: gradient 6s ease infinite;
            border-radius: 10px;
            margin: 10px auto;
            width: 70%;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }

        .subtitle {
            text-align: center;
            font-size: 1.2em;
            color: #4D4D4D;
            margin: 20px auto;
            padding: 5px;
            border-left: 4px solid #FF9F40;
            background-color: #FFF7E6;
            border-radius: 5px;
            width: 80%;
        }

        .separator {
            border: 0;
            height: 1px;
            background: linear-gradient(to right, #FF9F40, #FFC048, #F77B91);
            margin: 30px auto;
            width: 80%;
        }
    </style>
    <div class="title">Diabetes Prediction App</div>
    <div class="subtitle">
        This app predicts whether a patient is diabetic based on their health data.
    </div>
    <hr class="separator">
    """,
    unsafe_allow_html=True
)


st.sidebar.header('Enter Patient Data')
st.sidebar.write("Please provide the following details for a diabetes checkup:")

def calc():
    pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=17, value=3)
    bp = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=122, value=70)
    bmi = st.sidebar.number_input('BMI', min_value=0, max_value=67, value=20)
    glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=200, value=120)
    skinthickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.4, value=0.47)
    insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=846, value=79)
    age = st.sidebar.number_input('Age', min_value=21, max_value=88, value=33)

    output = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'bp': bp,
        'skinthickness': skinthickness,
        'insulin': insulin,
        'bmi': bmi,
        'dpf': dpf,
        'age': age
    }
    report_data = pd.DataFrame(output, index=[0])
    return report_data

user_data = calc()

st.subheader('Patient Data Summary')
st.write(user_data)

x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

progress = st.progress(0)
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
progress.progress(100)

result = rf.predict(user_data)

st.subheader('Prediction Result:')
output = 'You are not Diabetic' if result[0] == 0 else 'You are Diabetic'
st.markdown(f"<h2 style='text-align: center; color: {'#4CAF50' if result[0] == 0 else '#FF4136'};'>{output}</h2>", unsafe_allow_html=True)

accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
st.subheader('Model Accuracy:')
st.write(f"{accuracy:.2f}%")