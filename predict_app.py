import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from kumoai.experimental import rfm
import time

# Load environment variables from .env file
load_dotenv()
KUMO_API_KEY = os.environ.get('KUMO_API_KEY')
# Ensure the API key is set before initializing the library
if KUMO_API_KEY:
    rfm.init(api_key=KUMO_API_KEY)
else:
    st.error("KUMO_API_KEY not found in environment variables. Please set it in a .env file.")
    st.stop()

# Load data files
try:
    actual = pd.read_pickle('data/ca_iso_actual.pkl')
    ahead1d = pd.read_pickle('data/ca_iso_1da.pkl')
    ahead2d = pd.read_pickle('data/ca_iso_2da.pkl')
    ahead7d = pd.read_pickle('data/ca_iso_7da.pkl')
    regions_df = pd.read_pickle('data/ca_iso_regions.pkl')
except FileNotFoundError:
    st.error("Data files not found. Please ensure 'ca_iso_actual.pkl', 'ca_iso_1da.pkl', 'ca_iso_2da.pkl', 'ca_iso_7da.pkl', and 'ca_iso_regions.pkl' are in a 'data' directory.")
    st.stop()

regions_list = sorted(regions_df['region'].unique())
id_to_region = regions_df['region'].to_dict()
region_to_id = {v: k for k, v in id_to_region.items()}

# Set up KumoRFM objects
regions = rfm.LocalTable(regions_df, name='regions')
regions.primary_key = 'region_id'
usage = rfm.LocalTable(actual, name='usage')
usage.time_column = 'time'
graph = rfm.LocalGraph(tables=[regions, usage])
graph.link(src_table="usage", fkey="region_id", dst_table="regions")
model = rfm.KumoRFM(graph)

def convert_to_datetime(year, month, day, hour):
    la_timezone = pytz.timezone('America/Los_Angeles')
    try:
        naive_dt = datetime(year=int(year), month=int(month), day=int(day), hour=int(hour), minute=0, second=0)
        localized_dt = la_timezone.localize(naive_dt)
        return localized_dt
    except:
        return False

def is_date_within_range(year, month, day, hour):
    la_timezone = pytz.timezone('America/Los_Angeles')
    
    start_date = la_timezone.localize(datetime(2022, 8, 1, 0, 0, 0))
    end_date = la_timezone.localize(datetime(2025, 7, 31, 23, 0, 0))

    try:
        input_dt_naive = datetime(year=int(year), month=int(month), day=int(day), hour=int(hour), minute=0, second=0)
        input_dt = la_timezone.localize(input_dt_naive)
        
        return start_date <= input_dt <= end_date
    except ValueError:
        return False

def single_prediction(hours_ahead, year, month, date, hour, region, model=model, region_to_id=region_to_id):
    region_id = region_to_id[region]
    query = f"PREDICT SUM(usage.usage, {hours_ahead-1}, {hours_ahead}, hours) FOR regions.region_id={region_id}"
    
    month = str(month).zfill(2)
    date = str(date).zfill(2)
    hour = str(hour).zfill(2)
    time_stamp = f'{year}-{month}-{date} {hour}:00:00'

    for attempt in range(1, 11):
        try:
            df = model.predict(query, anchor_time=pd.Timestamp(time_stamp))
            return df.iloc[0]['TARGET_PRED']
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < 10:
                print("Retrying...")
                time.sleep(0.1)
            else:
                print("All 10 attempts failed.")
                raise RuntimeError("Failed to get a successful prediction after 10 retries.")


def hourly_predictions(year, month, date, hour, region):
    hourly = {}
    for i in range(1, 25):
        prediction = single_prediction(i, year, month, date, hour, region)
        while prediction<0 or prediction is None:
            prediction = single_prediction(i, year, month, date, hour, region)
        hourly[i] = prediction
    
    return hourly

def get_historical_data_for_plot(df, timestamp_to_match, region_id):
    """
    Fetches usage data from a historical DataFrame (actual or benchmarks)
    for a specific timestamp and region_id.
    Assumes df['time'] column is already timezone-aware.
    """
    # Ensure timestamp_to_match is timezone-aware for robust comparison
    # (It should be coming from convert_to_datetime which localizes)
    
    filtered_data = df[(df['region_id'] == region_id) & (df['time'] == timestamp_to_match)]['usage']
    
    if not filtered_data.empty:
        return filtered_data.iloc[0]
    return np.nan

# --- Streamlit App UI ---
st.title('Hourly Electricity Consumption Prediction (MW)')
st.markdown("Select a date, time, and region to get 24-hour electricity consumption predictions. \n\n **Input date must be between August 1, 2022, 00:00:00 and July 31, 2025, 23:00:00.**")

# Input widgets in columns for better layout
col1, col2, col3 = st.columns(3)
col1, col2, col3 = st.columns(3)
with col1:
    year_options = list(range(2022, 2026))
    year_default_index = year_options.index(2025)
    year = st.selectbox('Year', options=year_options, index=year_default_index)

    date_options = list(range(1, 32))
    date_default_index = date_options.index(31)
    date = st.selectbox('Day', options=date_options, index=date_default_index)
with col2:
    month_options = list(range(1, 13))
    month_default_index = month_options.index(7)
    month = st.selectbox('Month', options=month_options, index=month_default_index)

    hour_options = list(range(0, 24))
    hour_default_index = hour_options.index(23)
    hour = st.selectbox('Hour', options=hour_options, index=hour_default_index)
with col3:
    region_default_index = regions_list.index('PGE')
    region = st.selectbox('Area', options=regions_list, index=region_default_index)


if st.button('Run Prediction'):
    if not convert_to_datetime(year, month, date, hour):
        st.error(f"Invalid date or time provided. Please check your selections.")
    if not is_date_within_range(year, month, date, hour):
        st.error("❌ The selected date and time are outside the allowed range (August 1, 2022, 00:00:00 to July 31, 2025, 23:00:00). Please adjust your selections.")
    else:
        st.subheader('Prediction Results')
        
        predictions = hourly_predictions(year, month, date, hour, region)

        if predictions:
            # Create a list of the next 24 hours
            start_time = convert_to_datetime(year, month, date, hour)
            if start_time:
                hours_list = [start_time + timedelta(hours=i) for i in range(1, 25)]
                
                # Prepare data for the table
                data = {
                    'Time': [h.strftime('%Y-%m-%d %H:00') for h in hours_list],
                    f'Predicted Electricity Usage for {region} in MW': [round(predictions[i+1]) for i in range(24)]
                }
                
                df_predictions = pd.DataFrame(data)
                st.dataframe(df_predictions, use_container_width=True, hide_index=True)
                st.success("✅ Predictions generated successfully!")



                st.subheader(f'Comparaison of Predicted Electricity Usage (MW) for {region}')
                
                plot_data = []
                region_id_val = region_to_id[region]

                for i, current_hour_dt in enumerate(hours_list):
                    kumo_pred_val = predictions.get(i + 1)

                    actual_val = get_historical_data_for_plot(actual, current_hour_dt, region_id_val)
                    ahead1d_val = get_historical_data_for_plot(ahead1d, current_hour_dt, region_id_val)
                    ahead2d_val = get_historical_data_for_plot(ahead2d, current_hour_dt, region_id_val)
                    ahead7d_val = get_historical_data_for_plot(ahead7d, current_hour_dt, region_id_val)

                    plot_data.append({
                        'Time': current_hour_dt,
                        'Actual': actual_val,
                        'Kumo Predicted': kumo_pred_val,
                        '1 Day Ahead (CA ISO)': ahead1d_val,
                        '2 Day Ahead (CA ISO)': ahead2d_val,
                        '7 Day Ahead (CA ISO)': ahead7d_val
                    })
                
                df_plot = pd.DataFrame(plot_data)
                df_plot = df_plot.set_index('Time')
                
                st.line_chart(df_plot)
                st.info("💡 Note: 'Actual' and 'Benchmark' values will only appear on the graph if historical data is available for the selected time range. Otherwise, they will show as gaps.")
