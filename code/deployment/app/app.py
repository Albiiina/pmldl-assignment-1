import requests
import streamlit as st


API_URL = "http://api:8000/predict"


st.set_page_config(page_title="Titanic Survival Prediction")
st.title("Titanic Survival Prediction")
st.write(
    "Enter the passenger information below to predict whether the passenger "
    "would have survived the Titanic disaster."
)

pclass = st.selectbox("Passenger class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
sibsp = st.number_input("Siblings or spouses aboard (SibSp)", 0, 10, 0)
parch = st.number_input("Parents or children aboard (Parch)", 0, 10, 0)
fare = st.number_input("Fare", min_value=0.0, value=0.0)
embarked = st.selectbox("Port of embarkation (Embarked)", ["C", "Q", "S"])

if st.button("Predict Survival"):
    passenger_data = {
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
    }

    try:
        response = requests.post(API_URL, json=passenger_data, timeout=10)
        response.raise_for_status()
        result = response.json()

        if result["survived"]:
            st.success("Prediction: Survived")
        else:
            st.warning("Prediction: Did not survive")

        if "survival_probability" in result:
            probability = result["survival_probability"] * 100
            st.write(f"Survival probability: {probability:.2f}%")

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the prediction API.")
    except requests.exceptions.RequestException as error:
        st.error(f"The prediction API returned an error: {error}")
    except (KeyError, TypeError, ValueError):
        st.error("The prediction API returned an invalid response.")
