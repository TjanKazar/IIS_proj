import joblib, pandas as pd

model = joblib.load("/home/tjan-kazar/storage/Faks/mag2/IIS/Projekt/iis/models/xgboost_traffic.pkl")
feature_cols = joblib.load("/home/tjan-kazar/storage/Faks/mag2/IIS/Projekt/iis/models/feature_cols.pkl")

importances = pd.Series(model.feature_importances_, index=feature_cols)
print(importances.sort_values(ascending=False))