from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import math
import os
from typing import Optional

from fastapi import UploadFile, File
from fastapi.staticfiles import StaticFiles
import shutil
# import pickle

app = FastAPI()

# Load the pre-trained model at startup
# with open("PJME_model.pkl", "rb") as f:
#     model = pickle.load(f)
model = xgb.XGBRegressor()
model.load_model("PJME_model.ubj") 
    
# Mount static folder for HTML/JS/CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the HTML page
@app.get("/")
async def read_root():
    return FileResponse("static/index_model.html")

# Handle file upload
@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    with open(f"Data/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Train the model after upload
    await train_model()
    return {"message": "File uploaded and model trained!"}

# --- Your existing functions ---
def create_features(df: pl.DataFrame) -> pl.DataFrame:
    for i in [1, 2, 24, 48, 168]:
        df = df.with_columns(pl.col("PJME_MW").shift(i).alias(f"lag_{i}"))
    df = df.drop_nulls()
    return df.with_columns([
        pl.col("Datetime").dt.hour().alias("hour"),
        pl.col("Datetime").dt.weekday().alias("dayofweek"),
        pl.col("Datetime").dt.quarter().alias("quarter"),
        pl.col("Datetime").dt.month().alias("month"),
        pl.col("Datetime").dt.year().alias("year"),
        pl.col("Datetime").dt.ordinal_day().alias("dayofyear"),
        pl.col("Datetime").dt.day().alias("dayofmonth"),
        pl.col("Datetime").dt.week().alias("week")
    ])

# --- Global variables for model and data ---
# model = None
FEATURES = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'week']
FEATURES += [f"lag_{i}" for i in [1, 2, 24, 48, 168]]
TARGET = 'PJME_MW'

# --- Endpoints ---
@app.get("/train")
async def train_model():
    global model
    try:
        df = pl.read_csv('Data/PJME_hourly.csv')
    except FileNotFoundError:
        df = pl.read_csv('/kaggle/input/pjm-hourly-energy-consumption-data/PJME_hourly.csv')
    df = df.with_columns(pl.col("Datetime").str.to_datetime()).sort("Datetime")
    df = create_features(df)
    split_date = pl.datetime(2015, 1, 1)
    train = df.filter(pl.col("Datetime") < split_date)
    test = df.filter(pl.col("Datetime") >= split_date)
    n = len(test)
    first_half_size = math.ceil(n / 2)
    val = test.slice(0, first_half_size)
    test = test.slice(first_half_size)
    X_train = train.select(FEATURES).to_numpy()
    y_train = train.select(TARGET).to_numpy().flatten()
    X_val, y_val = val.select(FEATURES).to_numpy(), val.select(TARGET).to_numpy().flatten()
    # reg = xgb.XGBRegressor(
    #     base_score=0.5,
    #     booster='gbtree',
    #     n_estimators=500,
    #     early_stopping_rounds=50,
    #     objective='reg:squarederror',
    #     max_depth=3,
    #     learning_rate=0.05,
    # )
    # reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=100)
    # print("Training is finished :)")
    # model = reg
    return {"status": "Model trained successfully"}

# @app.get("/predict")
# def predict():
#     # Use the pre-loaded model
#     return {"prediction": model.predict(X_test).tolist()}
    
@app.get("/predict")
async def predict():
    if model is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model not trained yet. Call /train first.")
    try:
        df = pl.read_csv('Data/PJME_hourly.csv')
    except FileNotFoundError:
        df = pl.read_csv('/kaggle/input/pjm-hourly-energy-consumption-data/PJME_hourly.csv')
    df = df.with_columns(pl.col("Datetime").str.to_datetime()).sort("Datetime")
    df = create_features(df)
    split_date = pl.datetime(2015, 1, 1)
    test = df.filter(pl.col("Datetime") >= split_date)
    n = len(test)
    first_half_size = math.ceil(n / 2)
    test = test.slice(first_half_size)
    X_test = test.select(FEATURES).to_numpy()
    test_preds = model.predict(X_test)
    test = test.with_columns(pl.Series(name="prediction", values=test_preds))
    actual = test.select(TARGET).to_numpy().flatten()
    preds = test.select("prediction").to_numpy().flatten()
    rmse = np.sqrt(mean_squared_error(actual, preds))
    mae = mean_absolute_error(actual, preds)
    mape = mean_absolute_percentage_error(actual, preds)
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "predictions": test_preds.tolist()
    }

@app.get("/visualize")
async def visualize():
    if model is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model not trained yet. Call /train first.")
    try:
        df = pl.read_csv('Data/PJME_hourly.csv')
    except FileNotFoundError:
        df = pl.read_csv('/kaggle/input/pjm-hourly-energy-consumption-data/PJME_hourly.csv')
    df = df.with_columns(pl.col("Datetime").str.to_datetime()).sort("Datetime")
    df = create_features(df)
    split_date = pl.datetime(2015, 1, 1)
    test = df.filter(pl.col("Datetime") >= split_date)
    n = len(test)
    first_half_size = math.ceil(n / 2)
    test = test.slice(first_half_size)
    test_preds = model.predict(test.select(FEATURES).to_numpy())
    test = test.with_columns(pl.Series(name="prediction", values=test_preds))
    full_dates = df.select("Datetime").to_numpy().flatten()
    full_actual = df.select(TARGET).to_numpy().flatten()
    test_dates = test.select("Datetime").to_numpy().flatten()
    test_preds_plot = test.select("prediction").to_numpy().flatten()
    plt.figure(figsize=(15, 5))
    plt.plot(full_dates, full_actual, label='Actual', alpha=0.5)
    plt.plot(test_dates, test_preds_plot, label='Prediction', color='red', alpha=0.8)
    plt.title('PJM Energy Consumption Forecast (XGBoost + Polars)')
    plt.xlabel('Date')
    plt.ylabel('MW')
    plt.legend()
    plt.savefig('forecast_results.png')
    plt.close()
    plt.figure(figsize=(15, 5))
    last_month_test = test.tail(24*30)
    plt.plot(last_month_test.select("Datetime").to_numpy().flatten(),
             last_month_test.select(TARGET).to_numpy().flatten(), label='Actual')
    plt.plot(last_month_test.select("Datetime").to_numpy().flatten(),
             last_month_test.select("prediction").to_numpy().flatten(), label='Prediction')
    plt.title('Zoomed View: Last Month of Testing')
    plt.legend()
    plt.savefig('forecast_zoomed.png')
    plt.close()
    plt.figure(figsize=(10, 8))
    sorted_idx = np.argsort(model.feature_importances_)
    plt.barh(np.array(FEATURES)[sorted_idx], model.feature_importances_[sorted_idx])
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png')
    plt.close()
    return {
        "message": "Visualizations saved as forecast_results.png, forecast_zoomed.png, feature_importance.png",
        "files": ["forecast_results.png", "forecast_zoomed.png", "feature_importance.png"]
    }

@app.get("/download/{filename}")
async def download_file(filename: str):
    if not os.path.exists(filename):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    return FileResponse(filename, media_type="image/png", filename=filename)
