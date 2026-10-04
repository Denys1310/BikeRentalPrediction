import joblib
from datetime import date

import pandas as pd
import streamlit as st
from tensorflow import keras


st.set_page_config(
    page_title="Bike Rental Prediction",
    page_icon="🚲",
    layout="wide",
)


# Load trained model and preprocessor
@st.cache_resource
def load_resources():
    model = keras.models.load_model("models/bike_rental_model.keras")
    preprocessor = joblib.load("models/preprocessor.pkl")

    return model, preprocessor


# Validate input values
def validate_input(data):
    errors = []

    if not 0 <= data["Hour"] <= 23:
        errors.append("Hour must be between 0 and 23.")

    if not 0 <= data["Humidity(%)"] <= 100:
        errors.append("Humidity must be between 0 and 100%.")

    if data["Wind speed (m/s)"] < 0:
        errors.append("Wind speed cannot be negative.")

    if data["Visibility (10m)"] < 0:
        errors.append("Visibility cannot be negative.")

    if data["Solar Radiation (MJ/m2)"] < 0:
        errors.append("Solar radiation cannot be negative.")

    if data["Rainfall(mm)"] < 0:
        errors.append("Rainfall cannot be negative.")

    if data["Snowfall (cm)"] < 0:
        errors.append("Snowfall cannot be negative.")

    return errors


# Prepare input in the same format used during training
def prepare_input(
    selected_date,
    hour,
    temperature,
    humidity,
    wind_speed,
    visibility,
    dew_point,
    solar_radiation,
    rainfall,
    snowfall,
    season,
    holiday,
    functioning_day,
):
    return pd.DataFrame(
        [
            {
                "Hour": hour,
                "Temperature(Celsius)": temperature,
                "Humidity(%)": humidity,
                "Wind speed (m/s)": wind_speed,
                "Visibility (10m)": visibility,
                "Dew point temperature(Celsius)": dew_point,
                "Solar Radiation (MJ/m2)": solar_radiation,
                "Rainfall(mm)": rainfall,
                "Snowfall (cm)": snowfall,
                "Seasons": season,
                "Holiday": holiday,
                "Functioning Day": functioning_day,

                # Features created from Date during training
                "Year": selected_date.year,
                "Month": selected_date.month,
                "DayOfWeek": selected_date.weekday(),
            }
        ]
    )


# Header
st.title("🚲 Bike Rental Prediction")

st.write(
    "Enter date, time and weather conditions to predict "
    "the expected number of rented bikes."
)


# Load model
try:
    model, preprocessor = load_resources()

except Exception as error:
    st.error(f"Failed to load model or preprocessor: {error}")
    st.stop()


# Prediction form
with st.form("prediction_form"):

    # Date and time
    st.subheader("📅 Date and time")

    col1, col2 = st.columns(2)

    with col1:
        selected_date = st.date_input(
            "Date",
            value=date(2018, 6, 1),
        )

    with col2:
        hour = st.slider(
            "Hour",
            min_value=0,
            max_value=23,
            value=12,
        )

    # Weather
    st.subheader("🌤 Weather conditions")

    col1, col2, col3 = st.columns(3)

    with col1:
        temperature = st.number_input(
            "Temperature (°C)",
            value=20.0,
            step=0.1,
        )

        humidity = st.slider(
            "Humidity (%)",
            min_value=0,
            max_value=100,
            value=50,
        )

        wind_speed = st.number_input(
            "Wind speed (m/s)",
            min_value=0.0,
            value=1.5,
            step=0.1,
        )

    with col2:
        visibility = st.number_input(
            "Visibility (10m)",
            min_value=0,
            value=2000,
            step=10,
        )

        dew_point = st.number_input(
            "Dew point temperature (°C)",
            value=10.0,
            step=0.1,
        )

        solar_radiation = st.number_input(
            "Solar Radiation (MJ/m²)",
            min_value=0.0,
            value=0.5,
            step=0.01,
        )

    with col3:
        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=0.0,
            step=0.1,
        )

        snowfall = st.number_input(
            "Snowfall (cm)",
            min_value=0.0,
            value=0.0,
            step=0.1,
        )

    # Additional conditions
    st.subheader("⚙️ Additional conditions")

    col1, col2, col3 = st.columns(3)

    with col1:
        season = st.selectbox(
            "Season",
            [
                "Winter",
                "Spring",
                "Summer",
                "Autumn",
            ],
        )

    with col2:
        holiday = st.selectbox(
            "Holiday",
            [
                "No Holiday",
                "Holiday",
            ],
        )

    with col3:
        functioning_day = st.selectbox(
            "Functioning Day",
            [
                "Yes",
                "No",
            ],
        )

    submitted = st.form_submit_button(
        "Predict rented bikes",
        use_container_width=True,
    )


# Prediction
if submitted:

    validation_data = {
        "Hour": hour,
        "Humidity(%)": humidity,
        "Wind speed (m/s)": wind_speed,
        "Visibility (10m)": visibility,
        "Solar Radiation (MJ/m2)": solar_radiation,
        "Rainfall(mm)": rainfall,
        "Snowfall (cm)": snowfall,
    }

    errors = validate_input(validation_data)

    if errors:
        for error in errors:
            st.error(error)

    else:
        try:
            # Create DataFrame with exactly the features
            # used during model training
            input_df = prepare_input(
                selected_date=selected_date,
                hour=hour,
                temperature=temperature,
                humidity=humidity,
                wind_speed=wind_speed,
                visibility=visibility,
                dew_point=dew_point,
                solar_radiation=solar_radiation,
                rainfall=rainfall,
                snowfall=snowfall,
                season=season,
                holiday=holiday,
                functioning_day=functioning_day,
            )

            # Apply the trained preprocessor
            processed_data = preprocessor.transform(input_df)

            # Make prediction
            prediction = model.predict(
                processed_data,
                verbose=0,
            )[0][0]

            # Bike count cannot be negative
            prediction = max(0, round(float(prediction)))

            # Result
            st.divider()

            st.subheader("📊 Prediction result")

            st.metric(
                label="Expected number of rented bikes",
                value=f"{prediction} 🚲",
            )

            # Show generated features
            with st.expander("Input data used for prediction"):
                st.dataframe(
                    input_df,
                    use_container_width=True,
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")