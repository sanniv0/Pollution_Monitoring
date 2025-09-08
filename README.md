# Pollution Prediction & Insights Dashboard

This project provides a comprehensive solution for analyzing and predicting pollution levels across various cities. It includes data exploration, preprocessing, model training, and a Streamlit-based interactive dashboard for visualizing predictions and insights.

## Project Structure

- `Dataset_Cities.csv`: The primary dataset containing pollution data for various cities.
- `app.py`: The Streamlit application that serves as the interactive dashboard.
- `data_exploration.ipynb`: Jupyter Notebook for initial data exploration and understanding.
- `model_training.ipynb`: Jupyter Notebook for training the pollution prediction model.
- `preprocessing.ipynb`: Jupyter Notebook for data cleaning and preprocessing steps.
- `model.pkl`: The pre-trained machine learning model used by the Streamlit application.
- `pollution_heatmap.html`: An HTML file containing an interactive pollution heatmap, displayed within the Streamlit app.
- `Screenshot 2025-09-07 100721.png`: Screenshot of the application.
- `Screenshot 2025-09-07 021923.png`: Screenshot of the application.
- `Screenshot 2025-09-07 021933.png`: Screenshot of the application.

## Setup and Installation

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/sanniv0/Pollution_Monitoring.git
    cd Pollution_Monitoring
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    source venv/bin/activate  # On macOS/Linux
    ```

3.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file is not provided in the current context, but it is a standard practice to include one. You may need to create it based on the imports in the `.py` and `.ipynb` files.)*

## Usage

### 1. Data Preprocessing and Exploration

-   Explore the data characteristics and distributions using the `1.data_exploration.ipynb` notebook.
-   Open and run the `2.preprocessing.ipynb` notebook to clean and prepare the `Dataset_Cities.csv`.

### 2. Model Training

-   Execute the cells in `3.model_training.ipynb` to train the machine learning model. This notebook will save the trained model as `model.pkl`.
-   Run the `4.evaluation.ipynb` notebook to evaluate the model's performance.

### 3. Running the Streamlit Dashboard

-   Ensure `model.pkl` and `pollution_heatmap.html` are in the same directory as `app.py`.
-   Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

-   The application will open in your web browser, typically at `http://localhost:8500`.

## Dashboard Features

The Streamlit dashboard (`app.py`) provides the following functionalities:

-   **City Selection**: Select a city from the sidebar to view its specific data and predicted pollution level.
-   **City Data & Prediction**: Displays the raw data for the selected city and the predicted pollution level.
-   **SHAP Insights**: Visualizes SHAP (SHapley Additive exPlanations) values to explain the contribution of each feature to the model's prediction.
-   **Pollution Heatmap**: An interactive map showing pollution distribution, loaded from `pollution_heatmap.html`.

## Screenshots

Here are some screenshots of the application in action:

-   **Dashboard Overview**:
    ![Dashboard Overview](Screenshot%202025-09-07%20100721.png)

-   **SHAP Insights Section**:
    ![SHAP Insights](Screenshot%202025-09-07%20021923.png)

-   **Pollution Heatmap Section**:
    ![Pollution Heatmap](Screenshot%202025-09-07%20021933.png)

## Future Enhancements

-   **Add More Cities**: Expand the dataset to include more cities for broader analysis.
-   **Enhance Model**: Experiment with different machine learning models to improve prediction accuracy.
-   **User Authentication**: Implement user authentication for secure access to the dashboard.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

