import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta

# --- Real Data Loader ---
def load_prediction_data(run_directory="runs"):
    """
    Scans the specified directory for prediction .parquet files, loads them,
    and combines them into a single DataFrame for the app.
    """
    # Define a mapping from the filename abbreviation to the user-friendly name
    horizon_map = {
        '7DA': '7-Day Ahead (7DA)',
        '2DA': '2-Day Ahead (2DA)',
        'DAM': 'Day-Ahead (DAM)'
    }

    # Find all prediction files in the specified directory
    search_pattern = os.path.join(run_directory, "predictions_*.parquet")
    prediction_files = glob.glob(search_pattern)

    if not prediction_files:
        return pd.DataFrame() # Return an empty DataFrame if no files are found

    all_data = []
    for f_path in prediction_files:
        try:
            # Extract metadata from the filename (e.g., "predictions_2DA_sample100_...")
            filename = os.path.basename(f_path)
            parts = filename.split('_')
            horizon_key = parts[1]
            sample_size_str = parts[2].replace('sample', '')

            # Read the parquet file
            df = pd.read_parquet(f_path)

            # Add the parsed metadata as new columns
            df['horizon'] = horizon_map.get(horizon_key, "Unknown Horizon")
            df['sample_size'] = int(sample_size_str)
            
            all_data.append(df)
        except (IndexError, ValueError) as e:
            # This handles cases where a filename doesn't match the expected pattern
            st.warning(f"Could not parse filename: {filename}. Skipping. Error: {e}")
            continue
            
    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)

# --- Metric Calculation Functions ---
def calculate_mae(actual, predicted):
    return np.mean(np.abs(predicted - actual))

def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((predicted - actual)**2))

def calculate_mape(actual, predicted):
    # Avoid division by zero
    actual, predicted = np.array(actual), np.array(predicted)
    # Filter out where actual is zero to prevent division by zero errors
    mask = actual != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="Forecast Accuracy Benchmark")


st.title("⚔️ Kumo vs. CAISO Forecast Accuracy Benchmark")
st.markdown("Use the controls in the sidebar to select a forecast horizon and sample size to compare model performance.")

# --- Load the real data ---
# The app will look for a 'runs' subdirectory in the same folder it's running from.
data = load_prediction_data('runs')

if data.empty:
    st.error(
        "**No prediction data found.**\n\n"
        "Please ensure that:\n"
        "1. You have run your benchmark script to generate prediction files.\n"
        "2. The prediction files are located in a subdirectory named `runs`.\n"
        "3. The files follow the naming pattern: `predictions_{HORIZON}_sample{SIZE}_{TIMESTAMP}.parquet`"
    )
    st.stop() # Stop the app from running further if there's no data

# --- Sidebar for User Selections ---
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Get unique values for dropdowns from the dataframe
    available_horizons = sorted(data['horizon'].unique())
    available_sizes = sorted(data['sample_size'].unique())

    # Create dropdown menus
    selected_horizon = st.selectbox(
        "Select Time Horizon",
        options=available_horizons,
        index=0
    )

    selected_size = st.selectbox(
        "Select Sample Size",
        options=available_sizes,
        index=len(available_sizes) - 1 # Default to the largest size
    )

# --- Main Panel for Displaying Results ---

# Filter data based on user selection
filtered_data = data[
    (data['horizon'] == selected_horizon) &
    (data['sample_size'] == selected_size)
]

if filtered_data.empty:
    st.warning("No data available for the selected combination. Please make another selection.")
else:
    # Extract series for metric calculation
    actual = filtered_data['actual_mw']
    kumo_pred = filtered_data['predicted_mw_kumo']
    caiso_pred = filtered_data['predicted_mw_benchmark']

    # --- Display Metrics ---
    st.header(f"Performance Metrics for {selected_horizon} (n={selected_size})")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚀 Kumo Model")
        kumo_mae = calculate_mae(actual, kumo_pred)
        kumo_rmse = calculate_rmse(actual, kumo_pred)
        kumo_mape = calculate_mape(actual, kumo_pred)
        
        # Calculate CAISO metrics to compute the delta
        caiso_mae = calculate_mae(actual, caiso_pred)
        caiso_rmse = calculate_rmse(actual, caiso_pred)
        caiso_mape = calculate_mape(actual, caiso_pred)

        # Calculate percentage improvement for comparison visuals
        mae_improvement_pct = ((caiso_mae - kumo_mae) / caiso_mae) * 100 if caiso_mae != 0 else 0
        rmse_improvement_pct = ((caiso_rmse - kumo_rmse) / caiso_rmse) * 100 if caiso_rmse != 0 else 0
        # For MAPE, calculate the difference in percentage points (p.p.)
        mape_improvement_pp = kumo_mape - caiso_mape

        st.metric(label="Mean Absolute Error (MAE)", value=f"{kumo_mae:.1f} MW", delta=f"{mae_improvement_pct:.2f}% vs CAISO", delta_color="normal")
        st.metric(label="Root Mean Squared Error (RMSE)", value=f"{kumo_rmse:.1f} MW", delta=f"{rmse_improvement_pct:.2f}% vs CAISO", delta_color="normal")
        st.metric(label="Mean Absolute Percentage Error (MAPE)", value=f"{kumo_mape:.2f}%", delta=f"{mape_improvement_pp:.2f} p.p. vs CAISO", delta_color="inverse")

    with col2:
        st.subheader("📉 CAISO Benchmark")
        # Display CAISO metrics without delta, as it's now on the Kumo side
        st.metric(label="Mean Absolute Error (MAE)", value=f"{caiso_mae:.1f} MW")
        st.metric(label="Root Mean Squared Error (RMSE)", value=f"{caiso_rmse:.1f} MW")
        st.metric(label="Mean Absolute Percentage Error (MAPE)", value=f"{caiso_mape:.2f}%")
    
    st.info("Note: For 'Delta' values, green indicates better performance by the Kumo model.")

    # --- Display Data Table (UPDATED SECTION) ---
    st.header("Prediction Data")
    st.markdown("Browse the raw prediction data for the selected sample below.")

    # Create a copy to avoid modifying the original filtered_data
    display_df = filtered_data.copy()

    # 1. Sort the DataFrame by timestamp
    display_df = display_df.sort_values(by='prediction_timestamp')

    # 2. Add new columns for percentage difference as floats (not strings)
    # Handle potential division by zero for the actual values
    display_df['Kumo % Difference'] = ((display_df['predicted_mw_kumo'] - display_df['actual_mw']) / display_df['actual_mw']) * 100
    display_df['Benchmark % Difference'] = ((display_df['predicted_mw_benchmark'] - display_df['actual_mw']) / display_df['actual_mw']) * 100

    # 3. Round predicted values to the nearest integer
    # Rounding is done before display, so .style.format() won't affect it.
    display_df['predicted_mw_kumo'] = display_df['predicted_mw_kumo'].round(0).astype('Int64')
    display_df['predicted_mw_benchmark'] = display_df['predicted_mw_benchmark'].round(0).astype('Int64')

    # 4. Format the timestamp
    display_df['prediction_timestamp'] = pd.to_datetime(display_df['prediction_timestamp']).dt.strftime('%Y-%m-%d %H:%M')

    # 5. Select and rename columns for display
    display_df = display_df.rename(columns={
        'prediction_timestamp': 'Timestamp',
        'actual_mw': 'Actual MW',
        'predicted_mw_kumo': 'Kumo Forecast MW',
        'predicted_mw_benchmark': 'CAISO Forecast MW'
    })

    # The final columns to display
    final_cols = [
        'Timestamp',
        'Actual MW',
        'Kumo Forecast MW',
        'Kumo % Difference',
        'CAISO Forecast MW',
        'Benchmark % Difference'
    ]

    # Reorder the DataFrame columns to match the new list
    display_df = display_df[final_cols]

    # 6. Display the DataFrame with `.style.format()` for a seamless table
    st.dataframe(
        display_df.reset_index(drop=True).style.format({
            'Actual MW': '{:.0f}',
            'Kumo % Difference': '{:.2f}%',
            'Benchmark % Difference': '{:.2f}%'
        }),
        use_container_width=True,
        height=400
    )