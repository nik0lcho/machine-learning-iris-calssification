# Iris Classification with Scikit-Learn 🌸

This project is a simple machine learning classification task using the famous [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris).  
We build a model that classifies iris species based on flower measurements using scikit-learn.

## 🧠 Features
- Data loading and preprocessing
- Train/test split
- Trained a classification model using Random Forest
- Evaluation with accuracy and confusion matrix

## 🚀 How to Run the Project
- Clone the repo
- Start a virtual environment
- Install the requirements
- Follow the project structure
- Populate the raw directory with the csv file from the iris dataset
- Run the notebook with `jupyter notebook notebooks/exploration.ipynb`

## 📁 Project Structure

```bash
iris-classification/
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── data.py
│   ├── features.py
│   └── model.py
├── README.md
└── requirements.txt
