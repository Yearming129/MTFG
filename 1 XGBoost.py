import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load data
file_path = r'F:\Python file\2020\Newdata_2020.csv'
data = pd.read_csv(file_path, encoding='utf-8')

# 2. Data preprocessing
# Round 1 (First-layer modeling)
X = data[['CLCD', 'long', 'lat', 'Wind', 'Tem', 'Sun', 'Sand', 'Clay', 'Silt', 'Slope',
          'Relief', 'Pre', 'NDVI', 'Night', 'GPP',
          'ET', 'Erosion', 'Disturb', 'Water', 'Rail', 'Road',
          'Rock', 'DEM', 'Aspect']]
y = data['Frozen2005']

# # Round 2 (Second-layer modeling)
# X = data[['r2000', 'long', 'lat', 'Wind', 'Sunlight', 'Sand', 'Clay', 'Silt', 'Slope',
#           'Relief', 'Precipitat', 'NDVI', 'Nightlight', 'GPP',
#           'ET', 'Erosion', 'Disturb', 'Dis_Water', 'Dis_Railwa', 'Dis_road',
#           'RockDepth', 'DEM', 'Aspect']]
# y = data['SHAP_Temerature.1']

# 3. Divide the training set and the testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. XGBoost modelling
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    learning_rate=0.03,
    n_estimators=1800,
    max_depth=10,
    reg_lambda=22,
    reg_alpha=0,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# 5. Model evaluation
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("XGBoost Regression")
print(f"Training set MSE: {mse_train:.4f}, R²: {r2_train:.4f}")
print(f"Testing set MSE: {mse_test:.4f}, R²: {r2_test:.4f}")

# 6. Save the model
model_path = r'F:\SHAP uncertainty\2020\xgb_Tem2020_uncertainty.json'
xgb_model.save_model(model_path)
print(f"Model is saved in: {model_path}")