# Environmental Data Analytics and Predictive Intelligence

## Overview
This project is an environmental monitoring and AQI prediction dashboard developed using Streamlit and Machine Learning. The dashboard analyzes environmental sensor data, visualizes historical AQI trends, monitors environmental conditions, and predicts future AQI values.

## Features
* Historical AQI Analysis
* Monthly, Hourly, Seasonal, and Rolling AQI Trends
* AQI Category Classification
* Rainfall Analysis
* Wind Speed Analysis
* Gas Sensor Monitoring
* Device Status Monitoring
* Alert Analysis
* AQI Anomaly Detection
* AQI Forecasting for Next 15 Days
* AQI Forecasting for Next 30 Days
* Interactive Device Filtering

## Technologies Used
* Python
* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib

## Machine Learning
Model Used:
* Random Forest Regressor

Target Variable:
* Env_AQI

Features:
* Env_CO2
* Env_CO
* Env_NO2
* Env_SO2
* Env_O3
* Env_PM2_5
* Env_PM10
* Env_Avg_Temp
* Env_Relativehumidity

## Project Structure
AQI-Dashboard/
│
├── app.py
├── environment_data_small.csv
├── status_data_small.csv
├── alert_data_small.csv
├── gas_data_small.csv
├── requirements.txt
└── README.md

## Installation
Install the required dependencies:
pip install -r requirements.txt

## Run the Application
streamlit run app.py
After running the command, open the generated local URL in a web browser.

## Dataset
The project uses environmental, gas sensor, device status and alert datasets stored in CSV format.

## Author
Maanyaa

Maanyaa
