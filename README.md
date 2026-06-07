# credit-card-fraud-detection-engine
Machine Learning project for detecting fraudulent credit card transactions.
# Credit Card Fraud Detection Engine

## Overview

This project is an end-to-end machine learning application for detecting potentially fraudulent credit card transactions. The workflow covers synthetic data generation, feature engineering, model training, evaluation, and deployment using Streamlit and Hugging Face Spaces.

## Features

* Synthetic transaction dataset generation
* Fraud pattern injection
* Data cleaning and preprocessing
* Feature engineering using velocity-based risk scoring
* Random Forest classification
* Streamlit web application
* Cloud deployment on Hugging Face Spaces

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Hugging Face Spaces

## Model Features

* Transaction Amount
* Hours Since Last Transaction
* Distance From Home
* Velocity Risk Score
* Unknown Device Indicator

## Results

The model was trained on a synthetic imbalanced fraud dataset and achieved high fraud detection performance on the test set.

## Deployment

Live Demo:(https://huggingface.co/spaces/Anna-369/credit-card-fraud-detection)

## Repository Structure

credit-card-fraud-detection-engine/
├── notebooks/
├── src/
├   ├──fraud_engine.pkl
├   ├── streamlit_app.py
├── requirements.txt
├── README.md
└── screenshots/
