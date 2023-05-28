from typing import List
import streamlit as st
import pandas as pd
import runsor.data
from config import config
from pathlib import Path
from runsor import utils, main
from frontend import data


def highlight_col(col) -> List:
    """

    :param col:
    :return:
    """
    if col.name == "Calories":
        color = "#630707"
    else:
        color = "#0e1117"
    return ['background-color: {}'.format(color) for _ in col]


# Setup HTML configuration
st.set_page_config(page_title="RunSor", initial_sidebar_state="expanded", page_icon=":running:")


def app():
    # Tittle
    st.title("RunSor · Running ML Project")

    # Sections
    st.header(":1234: Data")
    # Load and display project data
    df = pd.read_csv(Path(config.DATA_DIR, "activity_log.csv"))
    st.text(f"Running trainings: {len(df)}")
    st.write(df)

    # Display best model metrics
    st.divider()
    st.header(":bar_chart: Model performance")
    performance_fp = Path(config.CONFIG_DIR, "performance.json")
    performance = utils.load_dict(performance_fp)
    st.write(performance)

    # Select the number of trainings to predict
    st.divider()
    st.header(":chart_with_upwards_trend: Inference")
    upload_method = st.radio(
        "How do you want to transfer data",
        ("Single Run", "Many trainings")
    )

    # Define run_id for prediction
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()

    # Predict calories value for single run
    if upload_method == "Single Run":
        col1, col2 = st.columns(2)
        # Display input boxes
        with col1:
            distance = st.number_input("Insert distance", value=1.00, step=0.1)
            heart_rate = st.number_input("Insert your average Hear Rate", value=145)
            elev_gain = st.number_input("Insert elevation gain (meters)", value=25)
        with col2:
            time = st.text_input("Insert time **accepted format - (%H:red[:]%M:red[:]%S)**", placeholder="00:15:26")
            run_cadence = st.number_input("Insert run cadence", value=176)
            elev_loss = st.number_input("Insert elevation loss (meters)", value=15)
        pace = st.text_input("Insert pace **accepted format - (%M:red[:]%S)**", placeholder="05:26")
        # Display button after all data has been entered
        if distance and time and heart_rate and run_cadence and elev_gain and elev_loss and pace:
            submit = st.button("Calculate calories!")
            if submit:
                try:
                    run = {"Distance": distance, "Time": data.time_conversion(time),
                           "Avg Run Cadence": run_cadence, "Avg HR": heart_rate,
                           "Avg Pace": data.pace_conversion(pace), "Elev Gain": elev_gain, "Elev Loss": elev_loss}
                    # Predict value for given data and display them
                    prediction = main.predict_value(run, run_id)
                    calories = prediction[0]['predicted_calories']
                    st.subheader(f':fire: You burned :red[{calories}] calories')
                # Raise error when values are not correct
                except ValueError:
                    st.write('**Cannot calculate calories, because of the invalid values**')

    else:
        uploaded_file = st.file_uploader("Choose a file with your running data :red[**(.json, .csv)**]",
                                         type=["csv", "json"])
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
                st.write(':red[**Wrong data was passed**]')


if __name__ == "__main__":
    app()
