import streamlit as st
import pandas as pd
import joblib

# Load the model
def load_model():
    try:
        return joblib.load(r'C:\Users\thape\PycharmProjects\appstream\appHeart\best_model.pkl')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Get user input
def get_user_input():
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
    ca = st.slider('Number of Major Vessels (0-3) Colored by Fluoroscopy', 0, 3, 1)
    thal = st.selectbox('Thalassemia', options=[1, 2, 3])

    features = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    return pd.DataFrame(features, index=[0])

def main():
    st.title('Heart Disease Prediction App')
    st.write('Please input patient data to predict heart disease.')

    user_input = get_user_input()

    st.subheader('Patient Data')
    st.write(user_input)

    if st.button('Predict'):
        model = load_model()
        if model:
            try:
                prediction = model.predict(user_input)
                prediction_proba = model.predict_proba(user_input)

                st.subheader('Prediction')
                heart_disease = 'Positive for Heart Disease' if prediction[0] == 1 else 'Negative for Heart Disease'
                st.write(heart_disease)

                st.subheader('Prediction Probability')
                st.write(prediction_proba)
            except Exception as e:
                st.error(f"Error in prediction: {e}")

if __name__ == '__main__':
    main()
