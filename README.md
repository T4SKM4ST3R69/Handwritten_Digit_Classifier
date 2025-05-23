# Handwritten_Digit_Classifier

This project demonstrates handwritten digit classification using logistic regression on the MNIST dataset. The model is built with `scikit-learn` and deployed with a simple `Streamlit` app for interactive predictions.

## Features

- Train a multinomial logistic regression classifier on MNIST
- Scaled input using `StandardScaler`
- Evaluate with test accuracy and classification report
- Save and load model using `joblib`
- Interactive digit prediction using Streamlit

## Requirements

Install all dependencies using:

```pip install -r requirements.txt```

requirements.txt contains:

- streamlit
- streamlit-drawable-canvas
- scikit-learn
- numpy
- pandas
- pillow
- opencv-python
- joblib
- matplotlib
- seaborn
- scipy

## Files

```hndwritten_log_reg.ipynb``` - Jupyter notebook for training the model

```logistic_regression_mnist_model.joblib```- Trained logistic regression model

```mnist_scaler.joblib``` - Feature scaler

```streamlit_app.py``` - Streamlit app for prediction

```requirements.txt``` - Required packages
