
## Requirements

- Python 3.7+
- pandas
- scikit-learn
- joblib
- matplotlib
- streamlit

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/bvp-valuations-model.git
    cd bvp-valuations-model
    ```

2. Install the required Python packages:
    ```bash
    pip install pandas scikit-learn joblib matplotlib streamlit
    ```

## Usage

### Train the Model

1. Place your input CSV file (`bvp_comps_052424.csv`) in the `input/` directory.

2. Run the training script:
    ```bash
    python train_model.py
    ```

3. The script will output the trained model (`random_forest_bvp_valuations.pkl`) and the plot (`predicted_vs_actual_plot.png`) to the `output/` directory.

### Run the Streamlit App

1. Ensure the trained model file (`random_forest_bvp_valuations.pkl`) is in the `output/` directory.

2. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

3. Open the provided URL in your web browser to interact with the app. Input the required metrics and get the predicted EV/Forward Revenue multiple and EV.

## Files

- **input/bvp_comps_052424.csv**: Input CSV file containing company data.
- **model/random_forest_bvp_valuations.pkl**: Trained Random Forest model.
- **model/predicted_vs_actual_plot.png**: Plot showing predicted vs actual EV/Forward Revenue multiples.
- **app.py**: Streamlit app for inputting company metrics and getting predictions.
- **train_model.py**: Script to train the Random Forest model and generate the plot.
- **README.md**: Project documentation.


## Acknowledgments

- Based on the BVP data and inspired by various financial modeling techniques.
- Developed using open-source tools and libraries.
