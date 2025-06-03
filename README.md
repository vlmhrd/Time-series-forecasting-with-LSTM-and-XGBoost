# Time-series-forecasting-with-LSTM-and-XGBoost

# Goal:
This project aims to forecast electricity demand using two different machine learning approaches: a deep learning-based LSTM (Long Short-Term Memory) model with attention, and a classical tree-based XGBoost regression model. The objective is to compare their performance and highlight their respective strengths and weaknesses on the same [dataset](https://www.kaggle.com/datasets/ashfakyeafi/pbd-load-history)

# Methodology and Model Comparison:

1. LSTM with Attention
Preprocessing: Data was scaled (MinMaxScaler) and formatted into sequential windows for LSTM input, with sequence length set to 14 time steps.
Model: A bidirectional LSTM network was constructed with batch normalization, dropout, and a custom attention layer, allowing the model to ‘focus’ on relevant time steps.
Training: Early stopping was implemented to avoid overfitting.
Evaluation: The model achieved a low Mean Absolute Error (MAE ≈ 0.01 after normalization) and was able to closely track the actual demand curve.
Strengths: Excels at learning temporal dependencies and capturing complex sequential patterns; especially effective with large datasets containing long-term trends.
Weaknesses: Training can be computationally intensive; more difficult to interpret; requires more data preprocessing and tuning.
![image](https://github.com/user-attachments/assets/55dfd158-4598-4074-aa32-c5de9f403f75)

2. XGBoost Regression
Preprocessing: Lagged features (previous 7 demand values) were engineered; standard tabular format used.
Model: XGBoost regressor trained on all engineered features.
Training: Hyperparameter tuning via RandomizedSearchCV.
Evaluation:
MAE: 104.08
RMSE: 177.28
R²: 0.9959 (very high, indicating excellent fit)
Optimized RMSE after tuning: 171.55
Predictions closely follow the actual demand, with some lag in sudden changes.
Strengths: Fast to train, robust to missing data, easy to tune, and provides feature importance for interpretation.
Weaknesses: May underperform on highly sequential data where deep learning models can exploit temporal structure more effectively.
![image](https://github.com/user-attachments/assets/94e3fcc1-a977-4709-baaa-87440557f1f4)


# Conclusions:

- Performance: Both models demonstrate strong predictive ability on this time-series forecasting task, with XGBoost being very competitive in terms of error metrics and requiring less complex data preparation.
- Model Choice:
LSTM is preferable if the task requires modeling long-term temporal dependencies, and if computational resources are available.
XGBoost is a strong baseline, easier to interpret, and quicker to deploy for structured tabular data with engineered features.
Visualization: Both models’ predictions overlay closely with actual demand, but LSTM may offer slight advantages for capturing subtle, sequential dependencies.


# Tools and Technologies:

- Programming Language: Python (Jupyter Notebook)
- Data Science Libraries: pandas, numpy, matplotlib, plotly, statsmodels
- Machine Learning: scikit-learn (MinMaxScaler, metrics), XGBoost
- Deep Learning: TensorFlow, Keras (LSTM, Bidirectional, Attention, Dense, Dropout)
- Dataset: Features include date, year, month, day, weekday, hour, demand, and temperature.
