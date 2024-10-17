# Heart Disease Prediction

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

## Overview
The **Heart Disease Prediction** project aims to predict the likelihood of heart disease in patients based on their medical attributes. This machine learning project utilizes classification algorithms to analyze a dataset of patient health metrics and predict whether a patient is at risk for heart disease. The goal is to provide a tool that can assist healthcare providers in early diagnosis and treatment planning.

## Features
- **Data Preprocessing**: Handles missing values, normalizes the data, and encodes categorical features for machine learning models.
- **Classification Models**: Implements multiple algorithms such as Logistic Regression, Random Forest, Support Vector Machines (SVM), and K-Nearest Neighbors (KNN).
- **Model Evaluation**: Evaluates models using performance metrics such as accuracy, precision, recall, F1-score, and AUC-ROC curve.
- **User Input**: Accepts patient data as input for prediction after training the model.
- **Model Comparison**: Compares different machine learning models to identify the best-performing one for heart disease prediction.

## Dataset
The dataset used in this project is the **Heart Disease UCI Dataset**, which contains information about patient health metrics and whether they have heart disease. Key features include:
- **Age**: Age of the patient.
- **Sex**: Gender of the patient (1 = male, 0 = female).
- **Chest Pain Type**: Types of chest pain (4 values).
- **Resting Blood Pressure**: Blood pressure value at rest.
- **Cholesterol**: Serum cholesterol level in mg/dl.
- **Fasting Blood Sugar**: Whether fasting blood sugar is > 120 mg/dl (1 = true, 0 = false).
- **Resting ECG**: Resting electrocardiographic results (0, 1, 2).
- **Max Heart Rate**: Maximum heart rate achieved.
- **Exercise Induced Angina**: Whether angina was induced by exercise (1 = yes, 0 = no).
- **Oldpeak**: ST depression induced by exercise relative to rest.
- **Slope of ST Segment**: Slope of the peak exercise ST segment.
- **Number of Major Vessels**: Number of major vessels colored by fluoroscopy (0–3).
- **Thalassemia**: Blood disorder type (3 = normal; 6 = fixed defect; 7 = reversible defect).

The dataset can be accessed from the UCI Machine Learning Repository [here](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - Pandas (for data manipulation)
  - NumPy (for numerical operations)
  - Scikit-learn (for machine learning models)
  - Matplotlib, Seaborn (for data visualization)
- **IDE**: Jupyter Notebook, PyCharm, or any Python-compatible IDE

## Setup Instructions

### Prerequisites
- Python 3.x
- Required Python libraries (listed in `requirements.txt`)

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/anrao0037/Heart-Disease-Prediction.git
   ```

2. **Install Required Libraries**:
   Navigate to the project directory and install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**:
   Download the heart disease dataset from the [UCI repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) and place it in the `data/` folder of the project directory.

## Usage
1. **Data Preprocessing**:
   - Run the `data_preprocessing.py` script to clean and preprocess the dataset. This will handle missing values, encode categorical variables, and scale the numerical features for machine learning algorithms.

2. **Model Training**:
   - Run `train_model.py` to train the machine learning models on the heart disease dataset:
   ```bash
   python train_model.py
   ```

3. **Prediction**:
   - Use the `predict.py` script to make predictions for new patient data. Input the necessary health metrics, and the script will output whether the patient is at risk for heart disease.
   ```bash
   python predict.py
   ```

4. **Evaluation**:
   - The trained models will be evaluated based on accuracy, precision, recall, F1-score, and the AUC-ROC curve. You can visualize the performance metrics for each model to identify the best performer.

## Model Performance
The models are evaluated using the following metrics:
- **Accuracy**: The percentage of correct predictions.
- **Precision**: The proportion of positive identifications that were actually correct.
- **Recall**: The proportion of actual positives that were correctly identified.
- **F1-Score**: The harmonic mean of precision and recall.
- **AUC-ROC**: The area under the Receiver Operating Characteristic curve, which measures the trade-off between true positive and false positive rates.

## Contributing
If you would like to contribute to this project, feel free to fork the repository and submit a pull request. Contributions can include bug fixes, adding new features, or improving model performance.

