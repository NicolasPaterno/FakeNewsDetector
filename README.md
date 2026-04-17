# Fake News Detector

A machine learning project to classify news articles as **FAKE** or **REAL** using Natural Language Processing techniques.

## About

This project applies supervised learning to the fake news detection problem — a relevant real-world challenge in the age of misinformation. It covers the full ML pipeline: data exploration, feature engineering, model training, and evaluation.

Built as a hands-on introduction to data analysis and machine learning.

## What it does

- Loads and explores a labeled news dataset (~500 articles)
- Visualizes class distribution and most frequent words per class
- Converts raw text into numerical features using **TF-IDF vectorization**
- Trains and compares four classifiers:
  - SGD Classifier (Passive-Aggressive)
  - Logistic Regression
  - Linear SVC
  - Naive Bayes
- Evaluates each model with cross-validation and a held-out test set
- Outputs a confusion matrix for the best model

## Tech stack

- Python, Jupyter Notebook
- pandas, NumPy, Matplotlib
- scikit-learn (TF-IDF, classifiers, cross-validation)

## Dataset

`data/news.csv` — contains news article titles, body text, and binary labels (FAKE / REAL).

## Getting started

```bash
pip install pandas numpy matplotlib scikit-learn
jupyter notebook main.ipynb
```

## Results

The best-performing model achieves ~93–96% accuracy on the test set, depending on the classifier. LinearSVC and Logistic Regression consistently outperform simpler baselines.

## Author

Nicolas Paterno — learning data science and ML.