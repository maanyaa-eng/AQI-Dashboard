import streamlit as st
import pyodbc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



# Database Connection

DRIVER = "ODBC Driver 17 for SQL Server"
SERVER =    "DrSaarthak\SQLEXPRESS"
DATABASE = "CentralDB"
USERNAME = "sa"
PASSWORD = "maanyaa@2003"

conn = pyodbc.connect(f"DRIVER={{{DRIVER}}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD};")

# Environment Table
@st.cache_data
def load_environment_data():
    query = "SELECT * FROM tblEnvironmentSensorTxn"
    return pd.read_sql(query, conn)

env_df = load_environment_data()

env_df["Env_ReceiveDateTime"] = pd.to_datetime(env_df["Env_ReceiveDateTime"])
env = env_df.copy()

env_df["Env_AQI"] = pd.to_numeric(env_df["Env_AQI"],errors="coerce")
env_df["Env_Rainfall"] = pd.to_numeric(env_df["Env_Rainfall"],errors="coerce")
env_df["Env_Wind_Speed"] = pd.to_numeric(env_df["Env_Wind_Speed"],errors="coerce")

env_df = env_df.dropna(subset=["Env_AQI"]).copy()

# Date Features
env_df["Month"] = env_df["Env_ReceiveDateTime"].dt.month_name()
env_df["Year"] = env_df["Env_ReceiveDateTime"].dt.year
env_df["Daily"] = env_df["Env_ReceiveDateTime"].dt.date
env_df["Hour"] = env_df["Env_ReceiveDateTime"].dt.hour

# AQI Category
def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy"
    else:
        return "Hazardous"
env_df["AQI_Category"] = env_df["Env_AQI"].apply(aqi_category)

# Rolling Mean
env_df = env_df.sort_values("Env_ReceiveDateTime")
env_df["AQI_Rolling_Mean"] = (env_df["Env_AQI"].rolling(5).mean())

# Season
env_df["Month_Num"] = (env_df["Env_ReceiveDateTime"].dt.month)

def season(month):
    if month in [12,1,2]:
        return "Winter"
    elif month in [3,4,5]:
        return "Summer"
    elif month in [6,7,8]:
        return "Monsoon"
    else:
        return "Post-Monsoon"
env_df["Season"] = env_df["Month_Num"].apply(season)

# AQI Change

env_df["AQI_Changed"] = (env_df.groupby("DeviceID")["Env_AQI"].diff())
aqi_jump = env_df[env_df["AQI_Changed"].abs() > 100]

# Device Status
@st.cache_data
def load_status_data():
    status_query = "SELECT * FROM tblAllDeviceStatus"
    return pd.read_sql(status_query,conn)

status_df = load_status_data()

status_df["StatusDateTime"] = pd.to_datetime(status_df["StatusDateTime"])

status_df["Previous_Status"] = (status_df.groupby("DeviceID")["DeviceStatus"].shift())

status_change = status_df[status_df["DeviceStatus"]!=status_df["Previous_Status"]]

# Alerts
@st.cache_data
def load_alert_data():
    alert_query = "SELECT * FROM Alert"
    return pd.read_sql(alert_query,conn)

alert_df = load_alert_data()
date_cols = ["Sent","ReceivedDateTime","InsertDateTime","AlertAckDateTime","AlertCloseDateTime"]

for col in date_cols:
    if col in alert_df.columns:
        alert_df[col] = pd.to_datetime(alert_df[col],errors="coerce")
        
alert_df["DeviceId"] = pd.to_numeric(alert_df["DeviceId"],errors="coerce")
aqi_jump["DeviceID"] = pd.to_numeric(aqi_jump["DeviceID"],errors="coerce")

alert_df["DeviceId"] = alert_df["DeviceId"].astype("Int64")
aqi_jump["DeviceID"] = aqi_jump["DeviceID"].astype("Int64")

alert_df = alert_df.dropna(subset=["AlertId","DeviceId","Sent"])
device_alert = (alert_df.groupby("DeviceId")["AlertId"].count())

alert_df = alert_df.sort_values("Sent")
aqi_jump = aqi_jump.sort_values("Env_ReceiveDateTime")

aqi_alert_change = pd.merge_asof(alert_df,aqi_jump,left_on="Sent",right_on="Env_ReceiveDateTime",left_by="DeviceId",right_by="DeviceID")
aqi_alert_change["Alert_Delay"] = (aqi_alert_change["Sent"]- aqi_alert_change["Env_ReceiveDateTime"])


# Gas Sensor Data
gas_query = """
SELECT *
FROM tblGasSensorTxn
WHERE Gas_ReceiveDateTime >= '2024-08-01'
AND Gas_ReceiveDateTime < '2024-09-01'
"""

df = pd.read_sql(gas_query, conn)

df["Gas_ReceiveDateTime"] = pd.to_datetime(df["Gas_ReceiveDateTime"])
df_numeric_cols = df.columns.drop("Gas_ReceiveDateTime")
df[df_numeric_cols] = (df[df_numeric_cols].apply(pd.to_numeric, errors="coerce"))
# Filling missing value with median for numeric columns
df = df.fillna(df.median())


#  Now we will filter the data from oct 2022 to jan 2023 for trend analysis and model training from table tblEnvironmentSensorTxn
trend_env_df = env_df[(env_df["Env_ReceiveDateTime"] >= "2022-10-01") & (env_df["Env_ReceiveDateTime"] < "2023-02-01")]

# Now we will remove missing values from aqi column as only 8 values are missing
trend_env_df = trend_env_df.dropna(subset=["Env_AQI"])

# Feature Engineering

trend_env_df["Month"] = (trend_env_df["Env_ReceiveDateTime"].dt.month)
trend_env_df["Hour"] = (trend_env_df["Env_ReceiveDateTime"].dt.hour)
trend_env_df["Daily"] = (trend_env_df["Env_ReceiveDateTime"].dt.date)
trend_env_df["Year"]=(trend_env_df["Env_ReceiveDateTime"].dt.year)

# Selecting Features will zero missing value which effect the AQI values

features=["Env_CO2","Env_CO","Env_NO2","Env_SO2","Env_O3","Env_PM2_5","Env_PM10","Env_Avg_Temp","Env_Relativehumidity"]

# Dividing values to X and Y features

X = trend_env_df[features]
y = trend_env_df["Env_AQI"]

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Random Forest Regressor

random_env = RandomForestRegressor(n_estimators=100,random_state=42)
random_env.fit(X_train, y_train)
random_pred=random_env.predict(X_test)

random_mae=mean_absolute_error(y_test, random_pred)
random_mse=mean_squared_error(y_test,random_pred)
random_rmse=np.sqrt(mean_squared_error(y_test,random_pred))
random_r2_score=r2_score(y_test,random_pred)

# Using Random Forest and do the prediction analysis of AQI for next 15 days

future_input = X.tail(15)
predicted_aqi = random_env.predict(future_input)
future_15 = pd.DataFrame({"Day": range(1,16),"Predicted_AQI": predicted_aqi})

# Using Random Forest and do the prediction analysis of AQI for next 30 days

future_input = X.tail(30)
future_pred = random_env.predict(future_input)
future_30 = pd.DataFrame({"Day": range(1,31),"Predicted_AQI": future_pred})







# AQI HISTORY DASHBOARD
st.set_page_config(page_title="AQI History Dashboard",layout="wide")

# TITLE
st.title(" Environmental AQI History Dashboard")
st.subheader("Historical AQI Monitoring and Analytics")


# SIDEBAR
st.sidebar.header("Filters")
device_list = sorted(env_df["DeviceID"].dropna().astype(int).unique())

selected_device = st.sidebar.selectbox("Select Device",device_list)

# FILTER DATA
filtered_env = env_df[env_df["DeviceID"] == selected_device]

aqi_data = filtered_env[(filtered_env["Env_AQI"].notnull()) &(filtered_env["Env_AQI"] > 0)]

if len(aqi_data) > 0:
    # KPI SECTION
    st.header("AQI Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average AQI",round(filtered_env["Env_AQI"].mean(), 2))
    col2.metric("Maximum AQI",filtered_env["Env_AQI"].max())
    col3.metric("Minimum AQI",filtered_env["Env_AQI"].min())
    col4.metric("Total Records",filtered_env.shape[0])

    # HISTORICAL AQI TREND
    st.header("Historical AQI Trend")
    history_chart = (filtered_env.set_index("Env_ReceiveDateTime")["Env_AQI"].resample("1H").mean())
    st.line_chart(history_chart)

    # ROLLING AQI TREND
    st.header("Rolling AQI Trend")
    rolling_chart = (filtered_env.set_index("Env_ReceiveDateTime")["AQI_Rolling_Mean"].resample("1H").mean())
    st.line_chart(rolling_chart)

    # MONTHLY AQI TREND
    st.header("Monthly AQI Trend")
    monthly_chart = filtered_env.groupby("Month")["Env_AQI"].mean()
    st.area_chart(monthly_chart)

    # HOURLY AQI TREND
    st.header("Hourly AQI Trend")
    hourly_chart = filtered_env.groupby("Hour")["Env_AQI"].mean()
    st.line_chart(hourly_chart)

    # SEASONAL AQI ANALYSIS
    st.header("Seasonal AQI Analysis")
    season_chart = filtered_env.groupby("Season")["Env_AQI"].mean()
    st.bar_chart(season_chart)
    
    # AQI CATEGORY DISTRIBUTION
    st.header("AQI Category Distribution")
    aqi_category_chart = filtered_env.groupby("AQI_Category").size()
    st.bar_chart(aqi_category_chart)
else:
    st.write("No AQI Data Available for this Device.")

# Rainfall Summary
st.header("Rainfall Summary")
rain_data = filtered_env[filtered_env["Env_Rainfall"].notnull()]
if len(rain_data) > 0:
    st.header("Rainfall Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Rainfall",round(rain_data["Env_Rainfall"].mean(),2))
    
    col2.metric("Maximum Rainfall",rain_data["Env_Rainfall"].max())

    col3.metric("Minimum Rainfall",rain_data["Env_Rainfall"].min())

    col4.metric("Rainfall Records",rain_data.shape[0])
    
    # Rainfall Average Trend
    st.header("Rainfall Trend")
    rain_chart = (rain_data.set_index("Env_ReceiveDateTime")["Env_Rainfall"].resample("1D").mean())
    st.line_chart(rain_chart)
else:
    st.write("No Rainfall Data Available for this Device.")


# Wind Speed Summary Analysis
st.header("Wind Speed Analysis")
wind_data = filtered_env[filtered_env["Env_Wind_Speed"].notnull().copy()]
if len(wind_data) > 0:
    col1, col2 = st.columns(2)
    col1.metric("Average Wind Speed",round(wind_data["Env_Wind_Speed"].mean(), 2))
    col2.metric("Maximum Wind Speed",wind_data["Env_Wind_Speed"].max())

    # Wind Speed Trend
    st.header("Wind Speed Trend")
    wind_chart = (wind_data.set_index("Env_ReceiveDateTime")["Env_Wind_Speed"].resample("1D").mean())
    st.line_chart(wind_chart)
else:
    st.write("No Wind Speed Data Available for this Device.")
    

# Gas Sensor Summary
st.header("Gas Sensor Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average CO2",round(df["Gas_CO2"].mean(), 2))
col2.metric("Average CO",round(df["Gas_CO"].mean(), 2))
col3.metric("Average NO2",round(df["Gas_NO2"].mean(), 2))
col4.metric("Average Gas AQI",round(df["Gas_AQI"].mean(), 2))

# Gas AQI Trend
st.header("Gas AQI Trend")
gas_aqi_trend = (df.set_index("Gas_ReceiveDateTime")["Gas_AQI"].resample("1H").mean())
st.line_chart(gas_aqi_trend)

#  Average of Gas Levels
st.header("Major Gas Levels")
gas_levels = (df[["Gas_CO2","Gas_CO","Gas_NO2","Gas_O3","Gas_SO2"]].mean())
st.bar_chart(gas_levels)

# Particulate Matter Levels
st.header("Particulate Matter Levels")
pm_levels = (df[["Gas_PM2_5","Gas_PM10"]].mean())
st.bar_chart(pm_levels)

# AQI ANOMALY DETECTION
if len(aqi_data) > 0:
    st.header("AQI Sudden Changes")
    device_anomaly = aqi_jump[aqi_jump["DeviceID"] == selected_device]
    st.write("Total Sudden Changes:", device_anomaly.shape[0])
    st.dataframe(device_anomaly[[ "Env_ReceiveDateTime","Env_AQI","AQI_Changed"]].head(100))

# DEVICE STATUS EVENTS
st.header("Device Online / Offline Events")
device_status_events = status_change[status_change["DeviceID"] == selected_device]
st.write("Total Status Changes:",device_status_events.shape[0])
st.dataframe(device_status_events[["StatusDateTime","Previous_Status","DeviceStatus"]].head(100))

# EVENT TO ALERT CORRELATION
st.header("Event to Alert Correlation")
device_event_alert = aqi_alert_change[aqi_alert_change["DeviceId"] == selected_device]
st.write("Total Event Alerts:",device_event_alert.shape[0])
important_cols = [col for col in["AlertId","Sent","AQI_Changed","Alert_Delay"]if col in device_event_alert.columns]
st.dataframe(device_event_alert[["AlertId","Sent","AQI_Changed","Alert_Delay"]].head(100))



# ALERT SUMMARY
st.header("Alerts Generated Per Device")
top_alerts = (device_alert.sort_values(ascending=False).head(20))
st.bar_chart(top_alerts)

# AQI FORECAST
st.header("AQI Forecast")

tab1, tab2 = st.tabs(["Next 15 Days","Next 30 Days"])

with tab1:
    st.line_chart(future_15.set_index("Day"))
    st.dataframe(future_15, hide_index=True)

with tab2:
    st.line_chart(future_30.set_index("Day"))
    st.dataframe(future_30, hide_index=True)


# RAW AQI DATA
st.header("AQI Historical Data")
st.dataframe(filtered_env[["Env_ReceiveDateTime","DeviceID","Env_AQI","AQI_Category"]].tail(100))