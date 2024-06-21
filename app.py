import streamlit as st
import pandas as pd
import joblib

def load_model():
    return joblib.load(r'C:\Users\thape\PycharmProjects\appstream\appHeart\best_model.pkl')

def main():
    st.title('Heart Disease Prediction')
    st.write('Please input patient data to predict heart disease.')

    # Assuming these are the features your model was trained on:
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])
    trestbps = st.number_input('Resting Blood Pressure', min_value=80, max_value=200, value=120)
    chol = st.number_input('Serum Cholesterol', min_value=100, max_value=500, value=197)
    fbs = st.radio('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
    restecg = st.selectbox('Resting Electrocardiographic Results', options=[0, 1, 2])
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=140)
    exang = st.radio('Exercise Induced Angina', options=[0, 1])
    oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.0, 1.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=[0, 1, 2])
    ca = st.slider('Number of Major Vessels (0-3) Colored by Flourosopy', 0, 3, 1)
    thal = st.selectbox('Thalassemia', options=[1, 2, 3])

    # Create DataFrame for model input
    features = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })

    if st.button('Predict'):
        model = load_model()
        prediction = model.predict(features)
        st.write('Prediction: ', 'Positive for Heart Disease' if prediction[0] == 1 else 'Negative for Heart Disease')

if __name__ == '__main__':
    main()
