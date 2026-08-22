import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Agriculture System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "uploaded_image": None,
    "weed_detected": None,
    "weed_name": None,
    "weed_confidence": None,
    "recommendation": None,
    "last_detection_time": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# DEMO / INTEGRATION SETTINGS
# ============================================================

USE_DEMO_DATA = True

# ThingSpeak settings
THINGSPEAK_CHANNEL_ID = ""
THINGSPEAK_READ_API_KEY = ""

# OpenWeather settings
OPENWEATHER_API_KEY = ""

# YOLO model path
YOLO_MODEL_PATH = "models/weed_model.pt"


# ============================================================
# SAMPLE / DEMO SENSOR DATA
# Replace this later with real ESP32 / ThingSpeak data.
# ============================================================

DEMO_SENSOR_DATA = {
    "temperature": 29.0,
    "humidity": 68.0,
    "soil_moisture": 42.0,
    "water_level": 75.0,
    "rain": False,
    "pump_status": "OFF",
}


# ============================================================
# HELPER: GET SENSOR DATA
# ============================================================

def get_demo_sensor_data():
    """Temporary demo values."""

    return DEMO_SENSOR_DATA.copy()


def get_thingspeak_data(channel_id, api_key):
    """
    Reads latest sensor data from ThingSpeak.

    Expected ThingSpeak fields:
    field1 = soil moisture
    field2 = temperature
    field3 = humidity
    field4 = water level
    field5 = rain
    field6 = pump status

    Adjust these mappings to match your actual ThingSpeak setup.
    """

    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json"

    try:
        response = requests.get(
            url,
            params={
                "api_key": api_key,
                "results": 10,
            },
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()
        feeds = data.get("feeds", [])

        if not feeds:
            return None

        latest = feeds[-1]

        def to_float(value, default=0.0):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        sensor_data = {
            "temperature": to_float(latest.get("field2")),
            "humidity": to_float(latest.get("field3")),
            "soil_moisture": to_float(latest.get("field1")),
            "water_level": to_float(latest.get("field4")),
            "rain": str(latest.get("field5", "0")).lower() in [
                "1",
                "true",
                "yes",
            ],
            "pump_status": latest.get("field6", "OFF"),
        }

        return sensor_data

    except requests.RequestException:
        return None


def get_sensor_data():
    """Use ThingSpeak when configured; otherwise use demo data."""

    if (
        not USE_DEMO_DATA
        and THINGSPEAK_CHANNEL_ID
        and THINGSPEAK_READ_API_KEY
    ):
        data = get_thingspeak_data(
            THINGSPEAK_CHANNEL_ID,
            THINGSPEAK_READ_API_KEY,
        )

        if data is not None:
            return data

    return get_demo_sensor_data()


# ============================================================
# WEATHER
# ============================================================

def get_demo_weather():
    return {
        "temperature": 29.0,
        "humidity": 68.0,
        "rain_probability": 20.0,
        "wind_speed": 8.0,
        "description": "Partly cloudy",
    }


def get_openweather_data(api_key, latitude, longitude):
    """
    Optional OpenWeather integration.

    Weather API integration can later be connected to the
    irrigation and spraying decision engines.
    """

    url = "https://api.openweathermap.org/data/2.5/weather"

    try:
        response = requests.get(
            url,
            params={
                "lat": latitude,
                "lon": longitude,
                "appid": api_key,
                "units": "metric",
            },
            timeout=10,
        )

        response.raise_for_status()
        data = response.json()

        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "rain_probability": 0,
            "wind_speed": data["wind"]["speed"],
            "description": data["weather"][0]["description"],
        }

    except (requests.RequestException, KeyError, TypeError):
        return None


def get_weather_data():
    """
    Uses real weather API only when configured.
    Otherwise returns demo data.
    """

    if not OPENWEATHER_API_KEY:
        return get_demo_weather()

    # Replace these with your actual farm coordinates.
    latitude = 28.6139
    longitude = 77.2090

    weather = get_openweather_data(
        OPENWEATHER_API_KEY,
        latitude,
        longitude,
    )

    return weather if weather else get_demo_weather()


# ============================================================
# IRRIGATION AI
# ============================================================

def predict_irrigation(sensor_data, weather_data, model_name):
    """
    PLACEHOLDER for your real LSTM / GRU / Transformer model.

    Later this function will:
        1. Prepare the sensor time-series.
        2. Load the selected trained model.
        3. Generate irrigation prediction.
        4. Return the model result.

    Current version uses simple demo logic.
    """

    soil_moisture = sensor_data["soil_moisture"]
    rain_probability = weather_data["rain_probability"]

    # Demo decision
    if soil_moisture < 30 and rain_probability < 60:
        required = True
    else:
        required = False

    if required:
        message = "Irrigation Required"
    else:
        message = "Irrigation Not Required"

    return {
        "model": model_name,
        "required": required,
        "message": message,
        "confidence": 0.0,  # Replace with real model confidence if available.
    }


# ============================================================
# WEED AI
# ============================================================

def run_weed_detection(image_file):
    """
    PLACEHOLDER / MODEL ADAPTER.

    Your friend's trained YOLO model should eventually be
    connected here.

    Expected output:
        {
            "weed_detected": True/False,
            "weed_name": "...",
            "confidence": 0.94
        }

    The current version returns a demo result so that the
    dashboard can be tested before the trained model exists.
    """

    # --------------------------------------------------------
    # Example future integration:
    #
    # from ultralytics import YOLO
    #
    # model = YOLO(YOLO_MODEL_PATH)
    # results = model(image_file)
    #
    # Parse results here.
    # --------------------------------------------------------

    return {
        "weed_detected": True,
        "weed_name": "Detected Weed",
        "confidence": 94.0,
    }


# ============================================================
# TREATMENT RECOMMENDATION
# ============================================================

def get_treatment_recommendation(
    weed_name,
    crop_name,
    weather_data,
):
    """
    Returns treatment information.

    IMPORTANT:
    Actual herbicide/dose values must come from verified
    agricultural guidance or product-label information.
    Do not invent them in code.
    """

    if not weed_name:
        return None

    rain_probability = weather_data["rain_probability"]

    recommendation = {
        "weed_name": weed_name,
        "crop": crop_name,
        "treatment": "Verified agricultural recommendation required",
        "dose": "To be populated from verified guidance",
        "spray_window": "To be determined",
        "duration": "To be determined",
        "weather_status": "Suitable",
    }

    if rain_probability >= 60:
        recommendation["weather_status"] = "Delay spraying due to weather"

    return recommendation


# ============================================================
# CHART DATA
# ============================================================

def create_demo_sensor_history():
    """
    Temporary history.

    Later replace this with historical ESP32/ThingSpeak data.
    """

    return pd.DataFrame(
        {
            "Time": [
                "10:00",
                "10:10",
                "10:20",
                "10:30",
                "10:40",
                "10:50",
                "11:00",
                "11:10",
            ],
            "Soil Moisture": [65, 61, 58, 54, 50, 47, 44, 42],
            "Temperature": [26, 27, 28, 28, 29, 29, 29, 29],
            "Humidity": [74, 73, 72, 71, 70, 69, 68, 68],
        }
    )


def create_demo_water_history():
    return pd.DataFrame(
        {
            "Time": [
                "10:00",
                "10:10",
                "10:20",
                "10:30",
                "10:40",
                "10:50",
                "11:00",
            ],
            "Water Used (L)": [8, 10, 7, 12, 9, 11, 6],
        }
    )


# ============================================================
# LOAD CURRENT DATA
# ============================================================

sensor_data = get_sensor_data()
weather_data = get_weather_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌱 Smart Agriculture")

st.sidebar.caption("Precision Agriculture Dashboard")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "💧 Irrigation",
        "🌿 Weed Detection",
        "🌦 Weather",
        "🧪 Recommendations",
        "🤖 AI Models",
    ],
)

st.sidebar.divider()

st.sidebar.subheader("⚙️ Settings")

crop_name = st.sidebar.selectbox(
    "Crop",
    [
        "Select crop",
        "Wheat",
        "Rice",
        "Maize",
        "Cotton",
        "Vegetables",
        "Other",
    ],
)

model_name = st.sidebar.selectbox(
    "Irrigation AI Model",
    [
        "LSTM",
        "GRU",
        "Transformer",
    ],
)

st.sidebar.divider()

if USE_DEMO_DATA:
    st.sidebar.warning("Demo data mode")


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "🏠 Overview":

    st.title("🌱 Smart Agriculture Dashboard")

    st.write(
        "Integrated monitoring for smart irrigation, "
        "weed detection, weather-aware decisions, and "
        "AI-based agricultural recommendations."
    )

    st.divider()

    # --------------------------------------------------------
    # CURRENT FIELD METRICS
    # --------------------------------------------------------

    st.header("📊 Current Field Status", anchor=False)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🌡 Temperature",
        f"{sensor_data['temperature']:.1f} °C",
    )

    c2.metric(
        "💧 Soil Moisture",
        f"{sensor_data['soil_moisture']:.1f} %",
    )

    c3.metric(
        "💦 Humidity",
        f"{sensor_data['humidity']:.1f} %",
    )

    c4.metric(
        "🪣 Water Level",
        f"{sensor_data['water_level']:.1f} %",
    )

    st.divider()

    # --------------------------------------------------------
    # QUICK STATUS
    # --------------------------------------------------------

    st.header("⚡ System Status", anchor=False)

    status1, status2, status3 = st.columns(3)

    if sensor_data["rain"]:
        status1.warning("🌧 Rain Detected")
    else:
        status1.success("☀️ No Rain Detected")

    if sensor_data["water_level"] < 20:
        status2.error("🚨 Low Water Level")
    else:
        status2.success("✅ Water Available")

    status3.info(
        f"Pump: {sensor_data['pump_status']}"
    )

    st.divider()

    # --------------------------------------------------------
    # IRRIGATION PREVIEW
    # --------------------------------------------------------

    irrigation_prediction = predict_irrigation(
        sensor_data,
        weather_data,
        model_name,
    )

    st.header("💧 Irrigation Prediction", anchor=False)

    if irrigation_prediction["required"]:
        st.warning(
            f"Model: {model_name} → "
            "Irrigation Required"
        )
    else:
        st.success(
            f"Model: {model_name} → "
            "Irrigation Not Required"
        )

    st.divider()

    # --------------------------------------------------------
    # WEATHER PREVIEW
    # --------------------------------------------------------

    st.header("🌦 Weather", anchor=False)

    w1, w2, w3 = st.columns(3)

    w1.metric(
        "Temperature",
        f"{weather_data['temperature']:.1f} °C",
    )

    w2.metric(
        "Rain Probability",
        f"{weather_data['rain_probability']:.0f} %",
    )

    w3.metric(
        "Wind Speed",
        f"{weather_data['wind_speed']:.1f} m/s",
    )


# ============================================================
# IRRIGATION PAGE
# ============================================================

elif page == "💧 Irrigation":

    st.title("💧 Smart Irrigation")

    st.write(
        "Sensor monitoring + AI prediction + weather-aware scheduling."
    )

    st.divider()

    # --------------------------------------------------------
    # SENSOR VALUES
    # --------------------------------------------------------

    st.header("Current Sensor Readings", anchor=False)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Soil Moisture",
        f"{sensor_data['soil_moisture']:.1f} %",
    )

    c2.metric(
        "Temperature",
        f"{sensor_data['temperature']:.1f} °C",
    )

    c3.metric(
        "Humidity",
        f"{sensor_data['humidity']:.1f} %",
    )

    c4.metric(
        "Water Level",
        f"{sensor_data['water_level']:.1f} %",
    )

    st.divider()

    # --------------------------------------------------------
    # AI MODEL
    # --------------------------------------------------------

    st.header("🧠 AI Irrigation Prediction", anchor=False)

    prediction = predict_irrigation(
        sensor_data,
        weather_data,
        model_name,
    )

    p1, p2, p3 = st.columns(3)

    p1.metric("Selected Model", model_name)

    if prediction["required"]:
        p2.error("IRRIGATION REQUIRED")
    else:
        p2.success("IRRIGATION NOT REQUIRED")

    p3.metric(
        "Pump Status",
        sensor_data["pump_status"],
    )

    st.divider()

    # --------------------------------------------------------
    # WEATHER-AWARE LOGIC
    # --------------------------------------------------------

    st.header("🌦 Weather-Aware Scheduling", anchor=False)

    rain_probability = weather_data["rain_probability"]

    if prediction["required"] and rain_probability >= 60:

        st.warning(
            "AI predicts irrigation is needed, but significant "
            "rain is expected. Irrigation should be postponed."
        )

    elif prediction["required"]:

        st.success(
            "AI predicts irrigation is needed and current "
            "weather conditions do not indicate major rainfall."
        )

    else:

        st.info(
            "Irrigation is currently not required according "
            "to the selected model."
        )

    st.divider()

    # --------------------------------------------------------
    # SOIL MOISTURE GRAPH
    # --------------------------------------------------------

    st.header("📈 Sensor History", anchor=False)

    history_df = create_demo_sensor_history()

    fig = px.line(
        history_df,
        x="Time",
        y=["Soil Moisture"],
        markers=True,
        title="Soil Moisture Over Time",
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Moisture (%)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # WATER USAGE
    # --------------------------------------------------------

    water_df = create_demo_water_history()

    fig2 = px.bar(
        water_df,
        x="Time",
        y="Water Used (L)",
        title="Water Usage Over Time",
    )

    fig2.update_layout(
        xaxis_title="Time",
        yaxis_title="Water Used (L)",
    )

    st.plotly_chart(
        fig2,
        use_container_width=True,
    )


# ============================================================
# WEED DETECTION PAGE
# ============================================================

elif page == "🌿 Weed Detection":

    st.title("🌿 AI Weed Detection")

    st.write(
        "Upload a crop image or capture a photo and analyze it "
        "using the weed-detection model."
    )

    st.divider()

    # --------------------------------------------------------
    # IMAGE INPUT
    # --------------------------------------------------------

    input_method = st.radio(
        "Image source",
        [
            "Upload from device",
            "Take a photo",
        ],
        horizontal=True,
    )

    uploaded_file = None

    if input_method == "Upload from device":

        uploaded_file = st.file_uploader(
            "Choose a crop image",
            type=["jpg", "jpeg", "png"],
        )

    else:

        uploaded_file = st.camera_input(
            "Take a crop photo"
        )

    # --------------------------------------------------------
    # PROCESS IMAGE
    # --------------------------------------------------------

    if uploaded_file is not None:

        st.session_state.uploaded_image = uploaded_file

        st.image(
            uploaded_file,
            caption="Crop Image",
            use_container_width=True,
        )

        st.divider()

        if st.button(
            "🔍 Analyze Image",
            use_container_width=True,
        ):

            with st.spinner("Analyzing image..."):

                result = run_weed_detection(
                    uploaded_file
                )

            st.session_state.weed_detected = (
                result["weed_detected"]
            )

            st.session_state.weed_name = (
                result["weed_name"]
            )

            st.session_state.weed_confidence = (
                result["confidence"]
            )

            st.session_state.last_detection_time = (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

            # Create recommendation
            st.session_state.recommendation = (
                get_treatment_recommendation(
                    result["weed_name"],
                    crop_name,
                    weather_data,
                )
            )

    # --------------------------------------------------------
    # DETECTION RESULT
    # --------------------------------------------------------

    if st.session_state.weed_detected is not None:

        st.divider()

        st.header("🤖 Detection Result", anchor=False)

        r1, r2 = st.columns(2)

        if st.session_state.weed_detected:

            r1.error("🌿 WEED DETECTED")

        else:

            r1.success("✅ NO WEED DETECTED")

        r2.metric(
            "Confidence",
            f"{st.session_state.weed_confidence:.1f} %",
        )

        st.write(
            f"Detected Class: "
            f"**{st.session_state.weed_name}**"
        )

        if st.session_state.last_detection_time:
            st.caption(
                "Last analysis: "
                f"{st.session_state.last_detection_time}"
            )


# ============================================================
# WEATHER PAGE
# ============================================================

elif page == "🌦 Weather":

    st.title("🌦 Weather Monitoring")

    st.write(
        "Weather information used for irrigation and spraying decisions."
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Temperature",
        f"{weather_data['temperature']:.1f} °C",
    )

    c2.metric(
        "Humidity",
        f"{weather_data['humidity']:.1f} %",
    )

    c3.metric(
        "Rain Probability",
        f"{weather_data['rain_probability']:.0f} %",
    )

    c4.metric(
        "Wind Speed",
        f"{weather_data['wind_speed']:.1f} m/s",
    )

    st.divider()

    st.subheader("Forecast Status")

    st.info(
        f"Current condition: {weather_data['description']}"
    )

    if weather_data["rain_probability"] >= 60:

        st.warning(
            "Rain probability is high. Consider postponing "
            "irrigation and spraying operations."
        )

    else:

        st.success(
            "No major rainfall restriction is currently indicated."
        )


# ============================================================
# RECOMMENDATIONS PAGE
# ============================================================

elif page == "🧪 Recommendations":

    st.title("🧪 Agricultural Recommendations")

    st.write(
        "This section combines AI results, field data and weather "
        "information to present farmer guidance."
    )

    st.divider()

    if st.session_state.recommendation is None:

        st.info(
            "Run weed detection first to generate a weed-treatment "
            "recommendation."
        )

    else:

        recommendation = st.session_state.recommendation

        st.header("🌿 Weed Treatment Recommendation", anchor=False)

        a1, a2 = st.columns(2)

        a1.metric(
            "Detected Weed",
            recommendation["weed_name"],
        )

        a2.metric(
            "Crop",
            recommendation["crop"],
        )

        st.write(
            f"**Treatment:** "
            f"{recommendation['treatment']}"
        )

        st.write(
            f"**Dose:** "
            f"{recommendation['dose']}"
        )

        st.write(
            f"**Recommended Spray Window:** "
            f"{recommendation['spray_window']}"
        )

        st.write(
            f"**Spray Duration:** "
            f"{recommendation['duration']}"
        )

        if recommendation["weather_status"] == "Suitable":

            st.success(
                "🌤 Current weather status does not indicate "
                "a major spraying restriction."
            )

        else:

            st.warning(
                f"🌧 {recommendation['weather_status']}"
            )

        st.divider()

        st.warning(
            "Treatment and dosage values must be populated from "
            "verified agricultural guidance/product labels."
        )


# ============================================================
# AI MODELS PAGE
# ============================================================

elif page == "🤖 AI Models":

    st.title("🤖 Irrigation AI Models")

    st.write(
        "Comparison interface for the deep-learning models "
        "used for irrigation prediction."
    )

    st.divider()

    # --------------------------------------------------------
    # Model comparison table
    # --------------------------------------------------------

    model_comparison = pd.DataFrame(
        {
            "Model": [
                "LSTM",
                "GRU",
                "Transformer",
            ],
            "Approach": [
                "Recurrent",
                "Recurrent",
                "Attention-based",
            ],
            "Status": [
                "To be integrated",
                "To be integrated",
                "To be integrated",
            ],
            "RMSE": [
                "-",
                "-",
                "-",
            ],
            "MAE": [
                "-",
                "-",
                "-",
            ],
        }
    )

    st.dataframe(
        model_comparison,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Current Model")

    st.info(
        f"Selected irrigation model: **{model_name}**"
    )

    st.write(
        "The trained models will be connected through the "
        "`predict_irrigation()` function."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Smart Agriculture Project • ESP32 + IoT + AI + Weather + Computer Vision"
)