# Naive Bayes Drug Classification

## Project Overview
This project showcases a custom implementation of the Naive Bayes classifier to classify drugs based on patient attributes. It is designed as an educational tool to understand the mechanics behind the Naive Bayes algorithm and apply it to a simple medical dataset.

## Dataset
The dataset includes medical profiles of patients, detailing attributes such as age, sex, blood pressure (BP), cholesterol levels, and individual sodium (Na) and potassium (K) levels. The goal is to predict the type of drug prescribed to a patient based on these attributes.

## Implementation
The Naive Bayes classifier was built from scratch using Python, with detailed attention to the calculation of prior and conditional probabilities. The project includes:

- Data preprocessing and handling categorical variables.
- Calculation of prior probabilities and conditional probabilities for each class.
- Application of Bayes' Theorem to predict drug types for new patient data.

## Results
The model achieved 100% accuracy on the training data, a result that is highlighted as an artifact of the small and simple dataset used for demonstration purposes. This outcome serves to illustrate the theoretical capabilities of the model under controlled conditions but may not reflect performance in more realistic, complex scenarios.

## Getting Started
To run this project, clone the repository and ensure that you have Python installed, along with the following libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install the necessary libraries using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn

