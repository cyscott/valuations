import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'input/bvp_comps_052424.csv'  # Update this to your CSV file path
data = pd.read_csv(file_path)

# Preprocess the data by converting percentages to numerical values and dropping unnecessary columns
data_cleaned = data.copy()
columns_to_convert = ['Efficiency', 'Revenue Growth Rate', 'Gross Margin', 'Rule of X']

for column in columns_to_convert:
    data_cleaned[column] = data_cleaned[column].str.rstrip('%').astype('float') / 100.0

data_cleaned['EV / Forward Revenue'] = data_cleaned['EV / Forward Revenue'].str.rstrip('x').astype('float')

# Select features and target variable
features = data_cleaned[['Efficiency', 'Revenue Growth Rate', 'Gross Margin', 'Rule of X']]
target = data_cleaned['EV / Forward Revenue']

# Fit the random forest regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(features, target)

# Predict and evaluate the model
rf_predictions = rf_model.predict(features)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(target, rf_predictions, alpha=0.7)
plt.plot([min(target), max(target)], [min(target), max(target)], color='red', linestyle='--')
plt.xlabel('Actual EV / Forward Revenue')
plt.ylabel('Predicted EV / Forward Revenue')
plt.title('Actual vs. Predicted EV / Forward Revenue')
plt.grid(True)
plot_file_path = 'model/random_forest_bvp_valuations_plot.png'
plt.savefig(plot_file_path)
plt.close()

# Save the trained Random Forest model to a file
model_file_path = 'model/random_forest_bvp_valuations.pkl'
joblib.dump(rf_model, model_file_path)

print(f"Model saved to {model_file_path}")
print(f"Plot saved to {plot_file_path}")
