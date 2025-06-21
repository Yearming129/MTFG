import pandas as pd
import xgboost as xgb
import os

# Define the file path
base_input_path = r'F:\MTFG\integrated data'  # Input data path
base_model_path = r'F:\Python file'  # Model path
output_path = r'F:\MTFG\predict'  # Output data path

# Define the file and model parameters for each year
data_info = {
    2000: {
        "file": os.path.join(base_model_path, "2000", "Newdata2000.csv"),
        "model": os.path.join(base_model_path, "2000", "xgb_2000_1.json"),
        "x_columns": ['r2000', 'long', 'lat', 'Wind', 'Temerature', 'Sunlight', 'Sand', 'Clay', 'Silt', 'Slope', 'Relief',
                      'Precipitat', 'NDVI', 'Nightlight', 'GPP', 'ET', 'Erosion', 'Disturb', 'Dis_Water', 'Dis_Railwa', 'Dis_road',
                      'RockDepth', 'DEM', 'Aspect'],
        "y_column": 'Frozen2000'
    },
    2005: {
        "file": os.path.join(base_model_path, "2005", "Newdata_2005.csv"),
        "model": os.path.join(base_model_path, "2005", "xgb_2005.json"),
        "x_columns": ['CLCD', 'long', 'lat', 'Wind', 'Tem', 'Sun', 'Sand', 'Clay', 'Silt', 'Slope', 'Relief',
                      'Pre', 'NDVI', 'Night', 'GPP', 'ET', 'Erosion', 'Disturb', 'Water', 'Rail', 'Road',
                      'Rock', 'DEM', 'Aspect'],
        "y_column": 'Frozen2005'
    },
    2010: {
        "file": os.path.join(base_model_path, "2010", "Newdata_2010.csv"),
        "model": os.path.join(base_model_path, "2010", "xgb_2010.json"),
        "x_columns": ['CLCD', 'long', 'lat', 'Wind', 'Tem', 'Sun', 'Sand', 'Clay', 'Silt', 'Slope', 'Relief',
                      'Pre', 'NDVI', 'Night', 'GPP', 'ET', 'Erosion', 'Disturb', 'Water', 'Rail', 'Road',
                      'Rock', 'DEM', 'Aspect'],
        "y_column": 'Frozen2005'
    },
    2015: {
        "file": os.path.join(base_model_path, "2015", "Newdata_2015.csv"),
        "model": os.path.join(base_model_path, "2015", "xgb_2015.json"),
        "x_columns": ['CLCD', 'long', 'lat', 'Wind', 'Tem', 'Sun', 'Sand', 'Clay', 'Silt', 'Slope', 'Relief',
                      'Pre', 'NDVI', 'Night', 'GPP', 'ET', 'Erosion', 'Disturb', 'Water', 'Rail', 'Road',
                      'Rock', 'DEM', 'Aspect'],
        "y_column": 'Frozen2005'
    },
    2020: {
        "file": os.path.join(base_model_path, "2020", "Newdata_2020.csv"),
        "model": os.path.join(base_model_path, "2020", "xgb_2020.json"),
        "x_columns": ['CLCD', 'long', 'lat', 'Wind', 'Tem', 'Sun', 'Sand', 'Clay', 'Silt', 'Slope', 'Relief',
                      'Pre', 'NDVI', 'Night', 'GPP', 'ET', 'Erosion', 'Disturb', 'Water', 'Rail', 'Road',
                      'Rock', 'DEM', 'Aspect'],
        "y_column": 'Frozen2005'
    }
}

os.makedirs(output_path, exist_ok=True)

# Traverse and process the data of each year
for year, info in data_info.items():
    print(f"Processing year {year}...")

    data = pd.read_csv(info['file'])
    x_columns = info['x_columns']
    y_column = info['y_column']

    if not set(x_columns + [y_column]).issubset(data.columns):
        print(f"Error: Missing required columns in {info['file']}. Skipping year {year}.")
        continue

    X = data[x_columns]
    y_true = data[y_column]

    model = xgb.Booster()
    model.load_model(info['model'])

    dmatrix = xgb.DMatrix(X)
    y_pred = model.predict(dmatrix)

    # Add the predicted values and residuals to the data table
    data[f'predict_MTFG_{year}'] = y_pred
    data[f'residual_error_{year}'] = y_pred - y_true  # Residual = Predicted value - True value

    # Output the result to a new CSV
    output_file = os.path.join(output_path, f"Newdata_{year}_predicted.csv")
    data.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Finished processing year {year}. Output saved to {output_file}.")