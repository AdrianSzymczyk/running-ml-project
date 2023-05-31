from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import runsor.data
from config import config
from frontend import data as frontdata
from runsor import data, main, utils


def highlight_col(col) -> List:
    """

    :param col:
    :return:
    """
    if col.name == "Calories":
        color = "#630707"
    else:
        color = "#0e1117"
    return ["background-color: {}".format(color) for _ in col]


# Setup HTML configuration
st.set_page_config(page_title="RunSor", initial_sidebar_state="expanded", page_icon=":running:")


def app():
    # Read data from csv file
    df = pd.read_csv(Path(config.DATA_DIR, "activity_log.csv"))
    # Tittle
    st.title("RunSor · Running ML Project")

    activities = ["Data display", "Analysis charts", "Model metrics", "Inference"]
    st.sidebar.title("Navigation")
    choices = st.sidebar.radio(" ", activities, label_visibility="collapsed")

    # ************************* Start Data display section ***************************
    if choices == "Data display":
        st.header(":1234: Data")
        # Load and display project data
        st.text(f"Running trainings: {len(df)}")
        st.write(df)
    # ************************* End Data display section ***************************

    # ************************* Start Analysis charts section ***************************
    if choices == "Analysis charts":
        # Transforming data to plot charts
        df = data.plots_transform(df)
        tab1, tab2, tab3 = st.tabs(["Avg Pace ", "Distance", "Calories vs Distance"])

        # Display of distribution of Average Pace chart
        plt.figure(figsize=(10, 6))
        plt.hist(df["Pace Range"], edgecolor="black", bins=15)
        plt.xlabel("Average Pace Range")
        plt.xticks(rotation=60)
        plt.ylabel("Frequency")
        tab1.subheader("A tab with distribution of Average Pace")
        tab1.pyplot(plt)

        # Display of distribution of Distance chart
        plt.figure(figsize=(10, 6))
        plt.hist(df["Distance"], edgecolor="black", color="orange", bins=15)
        plt.xlabel("Average Distance Range")
        plt.ylabel("Frequency")
        tab2.subheader("A tab with distribution of Distance")
        tab2.pyplot(plt)

        plt.figure(figsize=(10, 7))
        plt.scatter(df["Calories"], df["Distance"], c="blue", alpha=0.5)
        plt.xlabel("Calories")
        plt.ylabel("Distance")
        plt.tight_layout()
        tab3.subheader("Relationship between Calories and Distance")
        tab3.pyplot(plt)
    # ************************* End Data display section ***************************

    # ************************* Start Model metrics section ***************************
    if choices == "Model metrics":
        # Display best model metrics
        st.divider()
        st.header(":bar_chart: Model performance")
        performance_fp = Path(config.CONFIG_DIR, "performance.json")
        performance = utils.load_dict(performance_fp)
        st.write(performance)
    # ************************* End Model metrics section ***************************

    # ************************* Start Inference section ***************************
    if choices == "Inference":
        # Select the number of trainings to predict
        st.divider()
        st.header(":chart_with_upwards_trend: Inference")
        upload_method = st.radio(
            "How do you want to transfer data", ("Single Run", "Many trainings")
        )

        # Define run_id for prediction
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()

        # Predict calories value for single run
        if upload_method == "Single Run":
            col1, col2 = st.columns(2)
            # Display input boxes
            with col1:
                distance = st.number_input("Insert distance [km]", value=1.00, step=0.1)
                heart_rate = st.number_input("Insert your average Hear Rate", value=145)
                elev_gain = st.number_input("Insert elevation gain [meters]", value=25)
            with col2:
                time = st.text_input(
                    "Insert time **accepted format - (%H:red[:]%M:red[:]%S)**",
                    placeholder="00:15:26",
                )
                run_cadence = st.number_input("Insert run cadence", value=176)
                elev_loss = st.number_input("Insert elevation loss [meters]", value=15)
            pace = st.text_input(
                "Insert pace [km/min] **accepted format - (%M:red[:]%S)**", placeholder="05:26"
            )
            # Display button after all data has been entered
            if (
                distance
                and time
                and heart_rate
                and run_cadence
                and elev_gain
                and elev_loss
                and pace
            ):
                submit = st.button("Calculate calories!")
                if submit:
                    try:
                        run = {
                            "Distance": distance,
                            "Time": frontdata.time_conversion(time),
                            "Avg Run Cadence": run_cadence,
                            "Avg HR": heart_rate,
                            "Avg Pace": frontdata.pace_conversion(pace),
                            "Elev Gain": elev_gain,
                            "Elev Loss": elev_loss,
                        }
                        # Predict value for given data and display them
                        prediction = main.predict_value(run, run_id)
                        calories = prediction[0]["predicted_calories"]
                        st.subheader(f":fire: You burned :red[{calories}] calories")
                    # Raise error when values are not correct
                    except ValueError:
                        st.write("**Cannot calculate calories, because of the invalid values**")

        else:
            uploaded_file = st.file_uploader(
                "Choose a file with your running data :red[**(.json, .csv)**]", type=["csv", "json"]
            )
            if uploaded_file is not None:
                # Get the file extension
                file_extension = uploaded_file.name.split(".")[-1]
                if file_extension == "csv":
                    dataframe = pd.read_csv(uploaded_file)
                else:
                    dataframe = pd.read_json(uploaded_file)
                try:
                    if len(dataframe.columns) != 7:
                        dataframe = runsor.data.preprocess(dataframe)
                    predictions = main.predict_value(dataframe, run_id)
                    # Append calculated calories to an empty list
                    calories: List = []
                    for run in predictions:
                        calories.append(run["predicted_calories"])
                    dataframe["Calories"] = calories
                    dataframe = dataframe.style.apply(highlight_col, axis=0)
                    st.subheader(":fire:*Calculated calories* ")
                    st.write(dataframe)
                except KeyError:
                    st.write(":red[**Wrong data was passed**]")
    # ************************* End Inference section ***************************


if __name__ == "__main__":
    app()
