
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT = Path(__file__).resolve().parent

MODEL_FILE = PROJECT / "models" / "tool_wear_model.pkl"

DATA_FILE = (
    PROJECT / "dataset" / "processed"
    / "milling_features_cleaned.csv"
)

REPORT_DIR = PROJECT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

SENSORS = [
    "smcAC",
    "smcDC",
    "vib_table",
    "vib_spindle",
    "AE_table",
    "AE_spindle"
]

ARTIFACT_VALUE = 2771.189697266

# Provisional screening limits used by the current app.
# Confirm these against the exact training-cleaning script
# before treating predictions as operationally reliable.
SENSOR_LIMITS = {
    "smcAC": 1e6,
    "smcDC": 1e6,
    "vib_table": 5,
    "vib_spindle": 5,
    "AE_table": 5,
    "AE_spindle": 5
}


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CNC Tool Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. PROFESSIONAL CSS DESIGN
# ============================================================

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #f3f6fb;
    color: #18243a;
    font-family: 'Segoe UI', sans-serif;
}

/* Main content width */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #101d35 0%,
        #172b4d 100%
    );
}

[data-testid="stSidebar"] * {
    color: #edf4ff !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: rgba(255,255,255,0.045);
    padding: 10px 12px;
    border-radius: 9px;
    margin-bottom: 5px;
}

/* Headings */
h1, h2, h3 {
    color: #152641;
    font-weight: 700;
}

h1 {
    letter-spacing: -1px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e4eaf3;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 4px 15px rgba(21, 38, 65, 0.045);
}

[data-testid="stMetricLabel"] {
    color: #68778f;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #153c75;
    font-weight: 750;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(
        90deg,
        #1769e0,
        #08a6c9
    );
    color: white;
    border: none;
    border-radius: 9px;
    padding: 0.65rem 1.2rem;
    font-weight: 650;
    transition: 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #1256bd,
        #0789a8
    );
    color: white;
    border: none;
    transform: translateY(-1px);
}

/* Input boxes */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    border-radius: 8px;
}

/* Section cards */
.section-card {
    background: white;
    border: 1px solid #e4eaf3;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 4px 15px rgba(21, 38, 65, 0.035);
}

/* Hero section */
.hero {
    background: linear-gradient(
        115deg,
        #102344 0%,
        #174b86 58%,
        #087f9f 100%
    );
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    margin-bottom: 8px;
}

.hero p {
    color: #d7e9ff;
    font-size: 16px;
    margin-bottom: 0;
}

/* Small labels */
.eyebrow {
    color: #65d8f1;
    font-size: 12px;
    font-weight: 750;
    letter-spacing: 2px;
}

/* Status */
.status-pill {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 30px;
    background: #dff8ed;
    color: #08764b;
    font-size: 12px;
    font-weight: 700;
}

/* Footer */
.footer {
    text-align: center;
    color: #8794a8;
    font-size: 12px;
    padding: 30px 0 10px 0;
}

/* Divider */
hr {
    border-color: #e4eaf3;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 4. LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model_bundle():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}"
        )

    return joblib.load(MODEL_FILE)


@st.cache_data
def load_training_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_FILE}"
        )

    return pd.read_csv(DATA_FILE)


try:
    bundle = load_model_bundle()
    training_data = load_training_data()

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

except Exception as error:

    st.error(f"Application loading error: {error}")
    st.stop()


# Preserve the material data type used during training.
materials = sorted(
    training_data["material"]
    .dropna()
    .unique()
    .tolist()
)

labelled_data = training_data.dropna(
    subset=["VB"]
).copy()


# ============================================================
# 5. COMMON FUNCTIONS
# ============================================================

def extract_sensor_features(sensor_df):
    """
    Extract seven statistical features from each sensor.
    """

    features = {}
    audit = []

    for sensor in SENSORS:

        values = pd.to_numeric(
            sensor_df[sensor],
            errors="coerce"
        ).to_numpy(dtype=float)

        limit = SENSOR_LIMITS[sensor]

        finite = np.isfinite(values)

        extreme = np.abs(values) > limit

        repeated = np.isclose(
            values,
            ARTIFACT_VALUE,
            rtol=1e-12,
            atol=1e-9
        )

        invalid = ~finite | extreme | repeated

        valid_values = values[~invalid]

        if len(valid_values) == 0:
            raise ValueError(
                f"No valid samples remain in {sensor}."
            )

        # Statistical features
        mean = np.mean(valid_values)
        std = np.std(valid_values)
        variance = np.var(valid_values)
        rms = np.sqrt(np.mean(valid_values ** 2))
        maximum = np.max(valid_values)
        minimum = np.min(valid_values)
        peak_to_peak = maximum - minimum

        features[f"{sensor}_mean"] = mean
        features[f"{sensor}_std"] = std
        features[f"{sensor}_rms"] = rms
        features[f"{sensor}_variance"] = variance
        features[f"{sensor}_max"] = maximum
        features[f"{sensor}_min"] = minimum
        features[f"{sensor}_peak_to_peak"] = peak_to_peak

        audit.append({
            "Sensor": sensor,
            "Total samples": len(values),
            "Valid samples": len(valid_values),
            "Excluded samples": int(invalid.sum())
        })

    return features, pd.DataFrame(audit)


def create_prediction_input(
    sensor_features,
    machining_time,
    doc,
    feed,
    material
):

    input_values = {
        **sensor_features,
        "time": machining_time,
        "DOC": doc,
        "feed": feed,
        "material": material
    }

    missing_features = [
        col for col in feature_columns
        if col not in input_values
    ]

    if missing_features:
        raise ValueError(
            "Missing model features: "
            + ", ".join(missing_features)
        )

    input_df = pd.DataFrame(
        [input_values]
    )[feature_columns]

    return input_df


def create_sensor_chart(sensor_df, sensor):

    chart_df = sensor_df[[sensor]].copy()

    chart_df["Sample"] = np.arange(
        len(chart_df)
    )

    fig = px.line(
        chart_df.iloc[::10],
        x="Sample",
        y=sensor,
        title=f"{sensor} Signal",
        template="plotly_white"
    )

    fig.update_traces(
        line=dict(color="#087fba", width=1.4)
    )

    fig.update_layout(
        height=330,
        margin=dict(l=15, r=15, t=55, b=15),
        xaxis_title="Sample index",
        yaxis_title="Amplitude",
        title_font_size=17
    )

    return fig


def create_prediction_gauge(value):

    maximum = max(
        1.0,
        float(labelled_data["VB"].max())
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={
                "suffix": " mm",
                "font": {"size": 34}
            },
            title={
                "text": "Estimated Flank Wear",
                "font": {"size": 18}
            },
            gauge={
                "axis": {
                    "range": [0, maximum]
                },
                "bar": {
                    "color": "#087fba"
                },
                "bgcolor": "#eaf0f8",
                "borderwidth": 0,
                "steps": [
                    {
                        "range": [0, maximum],
                        "color": "#eaf0f8"
                    }
                ]
            }
        )
    )

    fig.update_layout(
        height=290,
        margin=dict(l=20, r=20, t=50, b=15)
    )

    return fig


# ============================================================
# 6. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding:15px 4px 25px 4px;">
            <div style="font-size:34px;">⚙️</div>
            <div style="font-size:22px;font-weight:800;">
                CNC INTELLIGENCE
            </div>
            <div style="font-size:11px;color:#83dff2;">
                SMART MANUFACTURING ANALYTICS
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Tool Wear Prediction",
            "Sensor Analysis",
            "Model Information"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown(
        """
        <div style="
            background:rgba(255,255,255,0.07);
            padding:14px;
            border-radius:10px;
            font-size:12px;
        ">
            <div style="color:#62d7ed;font-weight:700;">
                SYSTEM STATUS
            </div>
            <div style="margin-top:8px;">
                ● ML model loaded
            </div>
            <div>
                ● NASA dataset loaded
            </div>
            <div>
                ● Prediction interface ready
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            position:relative;
            padding-top:35px;
            color:#a8bbd4;
            font-size:11px;
        ">
            M.Tech CIM Project<br>
            CNC Tool Wear Prediction<br>
            Python + Machine Learning
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 7. OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">
                SMART MANUFACTURING • PREDICTIVE ANALYTICS
            </div>
            <h1>CNC Tool Intelligence Platform</h1>
            <p>
                Machine learning-based estimation of CNC
                milling tool flank wear using multi-sensor
                signals and machining parameters.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total NASA Records",
        len(training_data)
    )

    c2.metric(
        "Measured Wear Records",
        len(labelled_data)
    )

    c3.metric(
        "Sensor Channels",
        len(SENSORS)
    )

    c4.metric(
        "Model Features",
        len(feature_columns)
    )

    st.write("")

    left, right = st.columns(
        [1.45, 1],
        gap="large"
    )

    with left:

        st.markdown(
            "### Flank Wear Distribution"
        )

        fig = px.histogram(
            labelled_data,
            x="VB",
            nbins=22,
            color_discrete_sequence=["#087fba"],
            template="plotly_white",
            labels={
                "VB": "Flank wear (mm)",
                "count": "Number of runs"
            }
        )

        fig.update_layout(
            height=390,
            margin=dict(l=15, r=15, t=20, b=20),
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.markdown(
            "### Wear Statistics"
        )

        stats = pd.DataFrame({
            "Metric": [
                "Mean wear",
                "Minimum wear",
                "Maximum wear",
                "Median wear",
                "Standard deviation"
            ],
            "Value (mm)": [
                labelled_data["VB"].mean(),
                labelled_data["VB"].min(),
                labelled_data["VB"].max(),
                labelled_data["VB"].median(),
                labelled_data["VB"].std()
            ]
        })

        stats["Value (mm)"] = stats[
            "Value (mm)"
        ].round(4)

        st.dataframe(
            stats,
            hide_index=True,
            use_container_width=True
        )

        st.markdown(
            """
            <div class="section-card">
                <b style="color:#1769e0;">
                    DATASET INFORMATION
                </b>
                <p style="margin-top:12px;">
                    NASA Milling Dataset
                </p>
                <p>
                    16 experimental cases
                </p>
                <p>
                    Six sensor channels per run
                </p>
                <p>
                    9,000 samples per sensor run
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown("### Experimental Case Overview")

    case_stats = (
        labelled_data.groupby("case")["VB"]
        .agg(["count", "mean", "min", "max"])
        .reset_index()
    )

    case_stats.columns = [
        "Case",
        "Measured runs",
        "Mean VB",
        "Minimum VB",
        "Maximum VB"
    ]

    case_stats = case_stats.round(4)

    st.dataframe(
        case_stats,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# 8. TOOL WEAR PREDICTION PAGE
# ============================================================

elif page == "Tool Wear Prediction":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">
                MACHINE LEARNING • LIVE ESTIMATION
            </div>
            <h1>Tool Wear Prediction</h1>
            <p>
                Upload CNC sensor signals and machining
                parameters to estimate flank wear.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 01 — Upload Sensor Data")

    uploaded_file = st.file_uploader(
        "Upload one machining run in CSV format",
        type=["csv"],
        help=(
            "Required columns: smcAC, smcDC, vib_table, "
            "vib_spindle, AE_table, AE_spindle"
        )
    )

    st.caption(
        "Required sensor columns: "
        + ", ".join(SENSORS)
    )

    sensor_df = None

    if uploaded_file is not None:

        try:

            sensor_df = pd.read_csv(uploaded_file)

            missing = [
                sensor for sensor in SENSORS
                if sensor not in sensor_df.columns
            ]

            if missing:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing)
                )

                sensor_df = None

            elif len(sensor_df) == 0:

                st.error("The uploaded CSV is empty.")
                sensor_df = None

            else:

                st.success(
                    f"Sensor file loaded: "
                    f"{len(sensor_df):,} rows"
                )

        except Exception as error:

            st.error(
                f"Could not read CSV: {error}"
            )

            sensor_df = None

    st.divider()

    st.markdown("### 02 — Machining Parameters")

    p1, p2, p3, p4 = st.columns(4)

    with p1:

        machining_time = st.number_input(
            "Machining time",
            min_value=0.0,
            value=7.0,
            step=0.5
        )

    with p2:

        doc = st.number_input(
            "Depth of cut (DOC)",
            min_value=0.0,
            value=1.5,
            step=0.1
        )

    with p3:

        feed = st.number_input(
            "Feed rate",
            min_value=0.0,
            value=0.5,
            step=0.05
        )

    with p4:

        if materials:

            material = st.selectbox(
                "Material",
                options=materials
            )

        else:

            material = None
            st.error("No material values available.")

    st.divider()

    st.markdown("### 03 — Generate Prediction")

    if st.button(
        "⚡  PREDICT TOOL WEAR",
        use_container_width=True,
        disabled=(sensor_df is None or material is None)
    ):

        try:

            with st.spinner(
                "Processing sensor signals and predicting wear..."
            ):

                sensor_features, audit = (
                    extract_sensor_features(sensor_df)
                )

                input_df = create_prediction_input(
                    sensor_features,
                    machining_time,
                    doc,
                    feed,
                    material
                )

                raw_prediction = float(
                    model.predict(input_df)[0]
                )

                # Physical constraint:
                # negative flank wear is impossible.
                prediction = max(
                    0.0,
                    raw_prediction
                )

                # Store current prediction in session
                st.session_state["prediction"] = prediction
                st.session_state["raw_prediction"] = raw_prediction
                st.session_state["audit"] = audit
                st.session_state["input_df"] = input_df
                st.session_state["sensor_df"] = sensor_df.copy()
                st.session_state["parameters"] = {
                    "time": machining_time,
                    "DOC": doc,
                    "feed": feed,
                    "material": material
                }

        except Exception as error:

            st.error(
                f"Prediction error: {error}"
            )

    # Display prediction result
    if "prediction" in st.session_state:

        prediction = st.session_state["prediction"]
        raw_prediction = st.session_state["raw_prediction"]
        audit = st.session_state["audit"]

        st.divider()

        st.markdown("## Prediction Results")

        result1, result2 = st.columns(
            [1, 1.25],
            gap="large"
        )

        with result1:

            st.plotly_chart(
                create_prediction_gauge(prediction),
                use_container_width=True
            )

        with result2:

            st.metric(
                "Estimated Flank Wear",
                f"{prediction:.4f} mm"
            )

            if raw_prediction < 0:

                st.error(
                    f"Raw model prediction: "
                    f"{raw_prediction:.4f} mm. "
                    "Negative wear is physically impossible. "
                    "The displayed value has been constrained "
                    "to zero. This is a model warning, not a "
                    "validated prediction."
                )

            else:

                st.success(
                    "The model returned a non-negative "
                    "flank-wear estimate."
                )

            st.caption(
                "This is a machine-learning estimate, "
                "not a direct physical measurement."
            )

            st.write(
                f"Raw model output: "
                f"**{raw_prediction:.5f} mm**"
            )

        st.markdown("### Sensor Cleaning Audit")

        st.dataframe(
            audit,
            hide_index=True,
            use_container_width=True
        )

        if audit["Excluded samples"].sum() > 0:

            st.warning(
                "Some sensor samples were excluded using "
                "the application's provisional screening rules. "
                "Verify that these rules match the training "
                "preprocessing exactly."
            )

        # Sensor graph
        st.markdown("### Sensor Signal Visualization")

        selected_sensor = st.selectbox(
            "Select sensor",
            SENSORS
        )

        st.plotly_chart(
            create_sensor_chart(
                st.session_state["sensor_df"],
                selected_sensor
            ),
            use_container_width=True
        )

        # Download report
        parameters = st.session_state["parameters"]

        report = pd.DataFrame([{
            "Date": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Predicted_VB_mm": prediction,
            "Raw_Prediction_mm": raw_prediction,
            "Machining_Time": parameters["time"],
            "DOC": parameters["DOC"],
            "Feed": parameters["feed"],
            "Material": parameters["material"],
            "Excluded_Sensor_Samples": int(
                audit["Excluded samples"].sum()
            )
        }])

        st.download_button(
            "⬇ Download Prediction Report",
            data=report.to_csv(index=False),
            file_name="tool_wear_prediction_report.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# 9. SENSOR ANALYSIS PAGE
# ============================================================

elif page == "Sensor Analysis":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">
                MULTI-SENSOR DATA ANALYTICS
            </div>
            <h1>Sensor Analysis</h1>
            <p>
                Explore the statistical characteristics
                of the six CNC sensor channels.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    selected_feature_sensor = st.selectbox(
        "Choose sensor feature group",
        SENSORS
    )

    sensor_feature_columns = [
        f"{selected_feature_sensor}_{feature}"
        for feature in [
            "mean",
            "std",
            "rms",
            "variance",
            "max",
            "min",
            "peak_to_peak"
        ]
    ]

    available_columns = [
        col for col in sensor_feature_columns
        if col in training_data.columns
    ]

    if available_columns:

        feature_data = training_data[
            available_columns
        ].describe().T.reset_index()

        feature_data = feature_data.rename(
            columns={"index": "Feature"}
        )

        st.dataframe(
            feature_data.round(5),
            hide_index=True,
            use_container_width=True
        )

        if (
            f"{selected_feature_sensor}_rms"
            in training_data.columns
        ):

            fig = px.scatter(
                training_data.dropna(subset=["VB"]),
                x=f"{selected_feature_sensor}_rms",
                y="VB",
                color="case",
                title=(
                    f"{selected_feature_sensor} RMS "
                    "versus Flank Wear"
                ),
                template="plotly_white"
            )

            fig.update_layout(height=430)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.warning(
            "Sensor feature columns were not found."
        )


# ============================================================
# 10. MODEL INFORMATION PAGE
# ============================================================

elif page == "Model Information":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">
                MODEL DETAILS • SYSTEM INFORMATION
            </div>
            <h1>Model Information</h1>
            <p>
                Technical details of the CNC tool-wear
                estimation system.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Machine Learning Configuration")

    info = pd.DataFrame({
        "Property": [
            "Project",
            "Dataset",
            "Algorithm",
            "Training records",
            "Input features",
            "Target variable",
            "Output unit"
        ],
        "Value": [
            "CNC Tool Wear Prediction",
            "NASA Milling Dataset",
            bundle.get(
                "model_name",
                "Ridge Regression"
            ),
            bundle.get(
                "training_records",
                len(labelled_data)
            ),
            len(feature_columns),
            "Flank Wear (VB)",
            "Millimetres (mm)"
        ]
    })

    st.dataframe(
        info,
        hide_index=True,
        use_container_width=True
    )

    st.markdown("### Input Sensor Channels")

    sensor_info = pd.DataFrame({
        "Sensor": SENSORS,
        "Measurement type": [
            "Spindle motor AC current",
            "Spindle motor DC current",
            "Table vibration",
            "Spindle vibration",
            "Table acoustic emission",
            "Spindle acoustic emission"
        ],
        "Features extracted": [7] * 6
    })

    st.dataframe(
        sensor_info,
        hide_index=True,
        use_container_width=True
    )

    st.markdown("### Model Validation Reference")

    validation = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R²"],
        "Grouped validation result": [
            "0.1053 mm",
            "0.1541 mm",
            "0.6478"
        ]
    })

    st.dataframe(
        validation,
        hide_index=True,
        use_container_width=True
    )

    st.warning(
        "These are earlier validation results, not a "
        "guarantee of prediction accuracy for every uploaded run. "
        "The saved model was subsequently fitted on all 146 "
        "labelled records."
    )

    st.markdown("### Important Limitations")

    st.write(
        "- The model uses a relatively small experimental dataset."
    )

    st.write(
        "- Prediction quality depends on sensor preprocessing "
        "and machining conditions being consistent with training."
    )

    st.write(
        "- Negative raw predictions indicate model limitations."
    )

    st.write(
        "- This application is an academic demonstration and "
        "is not a certified machine-safety system."
    )


# ============================================================
# ============================================================
# 11. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        CNC TOOL INTELLIGENCE PLATFORM
        <br>
        Machine Learning • Smart Manufacturing • Predictive Analytics
        <br>
        Academic M.Tech CIM Project
        <br><br>
        Developed by <b>Aashu Prakash</b>
        <br>
        <a href="https://aashuprakash.vercel.app/"
           target="_blank"
           style="
               color: #087fba;
               font-weight: 700;
               text-decoration: none;
           ">
            🌐 Aashu Prakash
        </a>
    </div>
    """,
    unsafe_allow_html=True
)