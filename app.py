import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn

# Load the trained Random Forest model
model_file_path = 'model/random_forest_bvp_valuations.pkl'  # Ensure this path matches where you saved the model
rf_model = joblib.load(model_file_path)

st.title('EV/Forward Revenue Multiple and EV Calculator')

st.header('Input Company Metrics')

# Input fields for company metrics
efficiency = st.number_input('Efficiency (%)', min_value=0.0, step=0.1, format="%.1f", value=3.1) / 100.0
revenue_growth_rate = st.number_input('Revenue Growth Rate (%)', min_value=-100.0, step=0.1, value=15.0, format="%.1f") / 100.0
gross_margin = st.number_input('Gross Margin (%)', min_value=0.0, max_value=100.0, step=0.1, value=75.0, format="%.1f") / 100.0
rule_of_x = st.number_input('Rule of X (%)', min_value=-100.0, step=0.1, value=43.7, format="%.1f") / 100.0
ending_arr = st.number_input('Ending ARR (in millions)', min_value=0.0, step=0.01, value=6.34, format="%.2f")

# Predict EV / Forward Revenue multiple and calculate EV
if st.button('Calculate'):
    input_data = pd.DataFrame({
        'Efficiency': [efficiency],
        'Revenue Growth Rate': [revenue_growth_rate],
        'Gross Margin': [gross_margin],
        'Rule of X': [rule_of_x]
    })
    
    ev_forward_revenue_multiple = rf_model.predict(input_data)[0]
    ev = ev_forward_revenue_multiple * ending_arr * 1e6  # Convert millions to actual value
    
    st.subheader('Results')
    st.write(f'EV / Forward Revenue Multiple: {ev_forward_revenue_multiple:.2f}x')
    st.write(f'Enterprise Value (EV): ${ev:,.2f}')

# Display the plot
st.image('model/random_forest_bvp_valuations_plot.png', caption='Actual vs. Predicted EV / Forward Revenue')