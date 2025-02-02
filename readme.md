
## Requirements

- Python 3.x
- Streamlit
- Pandas
- XGBoost

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501` to interact with the application.

## Application Details

- The application uses a pre-trained XGBoost model stored in [xgb_model3.json](http://_vscodecontentref_/4).
- The user inputs various features related to health and lifestyle, which are then used to predict the risk of Alzheimer's disease.

## Model Features

The model expects the following features:

- Age
- Education Level
- BMI
- Diabetes
- Hypertension
- Cholesterol Level
- Family History of Alzheimer’s
- Cognitive Test Score
- Depression Level
- Sleep Quality
- Air Pollution Exposure
- Genetic Risk Factor (APOE-ε4 allele)
- Social Engagement Level
- Income Level
- Stress Levels
- Gender_Male
- Physical Activity Level_1
- Physical Activity Level_2
- Smoking Status_Former
- Smoking Status_Never
- Alcohol Consumption_Occasionally
- Alcohol Consumption_Regularly
- Dietary Habits_Healthy
- Dietary Habits_Unhealthy
- Employment Status_Retired
- Employment Status_Unemployed

## Prediction

- The user inputs are converted into a DataFrame and reindexed to match the model columns.
- The data is then converted into a DMatrix for XGBoost.
- The model predicts the risk of Alzheimer's disease, and the result is displayed on the Streamlit interface.

## License

This project is licensed under the MIT License.

## Acknowledgements

- The application uses [Streamlit](https://streamlit.io/) for the web interface.
- The model is built using [XGBoost](https://xgboost.readthedocs.io/).
