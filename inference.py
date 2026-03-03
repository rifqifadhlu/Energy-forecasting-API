import joblib
import pandas as pd

HORIZON =24

model = joblib.load("forecast_model.pkl")

def generates_features_from_history(history_df):
    last_timestamp = history_df.index[-1]

    df =history_df.copy()

    row = {}

    row['hour'] = last_timestamp.hour
    row['dayofweek'] = last_timestamp.dayofweek
    row['month'] = last_timestamp.month
    row['dayofyear'] = last_timestamp.dayofyear

    row['lag_1'] = df['DAYTON_MW'].iloc[-1]
    row['lag_24'] = df['DAYTON_MW'].iloc[-24]
    row['lag_168'] = df['DAYTON_MW'].iloc[-168]

    row['rolling_mean_24'] = df['DAYTON_MW'].iloc[-24].mean()
    row['rolling_std_24'] = df['DAYTON_MW'].iloc[-24].std()

    X_input = pd.DataFrame([row])
    return X_input


def predict_next_24_hours(history_df):
    X_input = generates_features_from_history(history_df)
    prediction = model.predict(X_input)[0]

    return prediction.tolist()