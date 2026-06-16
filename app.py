import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
# ----------------------------------
# Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="Student Mental Health Analytics Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------------
# Load Data
# ----------------------------------

df = pd.read_csv("data/Student Mental health.csv")

model = joblib.load("models/depression_model.pkl")
encoders = joblib.load("models/encoders.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")

# Clean Year column
df["Your current year of Study"] = (
    df["Your current year of Study"]
    .astype(str)
    .str.strip()
    .str.title()
)

# Handle missing ages
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

# ----------------------------------
# Sidebar Filters
# ----------------------------------

st.sidebar.header("Dashboard Filters")

course_filter = st.sidebar.selectbox(
    "Select Course",
    ["All"] + sorted(df["What is your course?"].dropna().unique().tolist())
)

year_filter = st.sidebar.selectbox(
    "Select Year",
    ["All"] + sorted(df["Your current year of Study"].dropna().unique().tolist())
)

gender_filter = st.sidebar.selectbox(
    "Select Gender",
    ["All"] + sorted(df["Choose your gender"].dropna().unique().tolist())
)

filtered_df = df.copy()

if course_filter != "All":
    filtered_df = filtered_df[
        filtered_df["What is your course?"] == course_filter
    ]

if year_filter != "All":
    filtered_df = filtered_df[
        filtered_df["Your current year of Study"] == year_filter
    ]

if gender_filter != "All":
    filtered_df = filtered_df[
        filtered_df["Choose your gender"] == gender_filter
    ]

st.sidebar.header("About")

st.sidebar.info("""
Student Mental Health Analytics Dashboard

Tech Stack:
• Python
• Streamlit
• Pandas
• Plotly
• Scikit-Learn

Model:
• Logistic Regression
• Accuracy: 85%
""")
# ----------------------------------
# Title
# ----------------------------------

st.title("🧠 Student Mental Health Analytics Dashboard")
st.markdown(
    "Analyze depression, anxiety, panic attacks, and treatment-seeking behavior among students."
)

# ----------------------------------
# KPI SECTION
# ----------------------------------

total_students = len(filtered_df)

depression_rate = (
    (filtered_df["Do you have Depression?"] == "Yes").sum()
    / total_students
) * 100

anxiety_rate = (
    (filtered_df["Do you have Anxiety?"] == "Yes").sum()
    / total_students
) * 100

panic_rate = (
    (filtered_df["Do you have Panic attack?"] == "Yes").sum()
    / total_students
) * 100

treatment_rate = (
    (
        filtered_df[
            "Did you seek any specialist for a treatment?"
        ] == "Yes"
    ).sum()
    / total_students
) * 100

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Students", total_students)
col2.metric("Depression", f"{depression_rate:.1f}%")
col3.metric("Anxiety", f"{anxiety_rate:.1f}%")
col4.metric("Panic Attack", f"{panic_rate:.1f}%")
col5.metric("Treatment", f"{treatment_rate:.1f}%")

st.divider()

st.divider()


st.header("🧠 Depression Risk Predictor")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    age = st.number_input(
        "Age",
        min_value=15,
        max_value=40,
        value=21
    )

with col2:
    year = st.selectbox(
        "Year of Study",
        ["Year 1", "Year 2", "Year 3", "Year 4"]
    )

with col3:
    cgpa = st.selectbox(
        "CGPA",
        [
            "0 - 1.99",
            "2.00 - 2.49",
            "2.50 - 2.99",
            "3.00 - 3.49",
            "3.50 - 4.00"
        ]
    )

marital = st.selectbox(
    "Marital Status",
    ["No", "Yes"]
)

if st.button("Predict Risk"):

    gender_encoded = encoders[
        "Choose your gender"
    ].transform([gender])[0]

    year_encoded = encoders[
        "Your current year of Study"
    ].transform([year])[0]

    cgpa_encoded = encoders[
        "What is your CGPA?"
    ].transform([cgpa])[0]

    marital_encoded = encoders[
        "Marital status"
    ].transform([marital])[0]

    features = np.array([
        [
            gender_encoded,
            age,
            year_encoded,
            cgpa_encoded,
            marital_encoded
        ]
    ])

    prediction = model.predict(features)[0]

    result = target_encoder.inverse_transform(
        [prediction]
    )[0]

    if result == "Yes":
        st.error("⚠️ Higher Depression Risk")
    else:
        st.success("✅ Lower Depression Risk")
# ----------------------------------
# Gender Distribution
# ----------------------------------

gender_counts = filtered_df["Choose your gender"].value_counts()

fig_gender = px.pie(
    values=gender_counts.values,
    names=gender_counts.index,
    title="Gender Distribution"
)

st.plotly_chart(fig_gender, use_container_width=True)

# ----------------------------------
# Age Distribution
# ----------------------------------

fig_age = px.histogram(
    filtered_df,
    x="Age",
    nbins=10,
    title="Age Distribution"
)

st.plotly_chart(fig_age, use_container_width=True)

# ----------------------------------
# Depression by Gender
# ----------------------------------

depression_gender = pd.crosstab(
    filtered_df["Choose your gender"],
    filtered_df["Do you have Depression?"]
)

fig_dep_gender = px.bar(
    depression_gender,
    barmode="group",
    title="Depression by Gender"
)

st.plotly_chart(fig_dep_gender, use_container_width=True)

# ----------------------------------
# Anxiety by Year
# ----------------------------------

anxiety_year = pd.crosstab(
    filtered_df["Your current year of Study"],
    filtered_df["Do you have Anxiety?"]
)

fig_anxiety_year = px.bar(
    anxiety_year,
    barmode="group",
    title="Anxiety by Year of Study"
)

st.plotly_chart(fig_anxiety_year, use_container_width=True)

# ----------------------------------
# Depression by Course
# ----------------------------------

depression_course = pd.crosstab(
    filtered_df["What is your course?"],
    filtered_df["Do you have Depression?"]
)

fig_dep_course = px.bar(
    depression_course,
    barmode="group",
    title="Depression by Course"
)

st.plotly_chart(fig_dep_course, use_container_width=True)

# ----------------------------------
# Treatment Seeking Analysis
# ----------------------------------

treatment_counts = filtered_df[
    "Did you seek any specialist for a treatment?"
].value_counts()

fig_treatment = px.bar(
    x=treatment_counts.index,
    y=treatment_counts.values,
    title="Treatment Seeking Analysis",
    labels={"x": "Response", "y": "Count"}
)

st.plotly_chart(fig_treatment, use_container_width=True)

# ----------------------------------
# CGPA Analysis
# ----------------------------------

cgpa_counts = filtered_df[
    "What is your CGPA?"
].value_counts()

fig_cgpa = px.bar(
    x=cgpa_counts.index,
    y=cgpa_counts.values,
    title="CGPA Distribution",
    labels={"x": "CGPA Range", "y": "Students"}
)

st.plotly_chart(fig_cgpa, use_container_width=True)

# ----------------------------------
# Download Dataset
# ----------------------------------

st.download_button(
    label="📥 Download Filtered Dataset",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_student_mental_health.csv",
    mime="text/csv"
)

# ----------------------------------
# Key Insights
# ----------------------------------

st.header("📌 Key Insights")

st.markdown("""
- Mental health concerns are present across multiple courses.
- Anxiety and depression are observed across different years of study.
- Not all affected students seek professional treatment.
- Academic and demographic factors may influence mental well-being.
""")

