import pandas as pd
import xgboost as xgb
import shap
import os
from tqdm import tqdm

def main():
    # 1. Load data
    file_path = r'F:\Python file\2020\Newdata_2020.csv'
    data = pd.read_csv(file_path, encoding='utf-8')

    # 2. Data preprocessing
    X = data[['CLCD', 'long', 'lat', 'Wind', 'Tem', 'Sun', 'Sand', 'Clay', 'Silt', 'Slope',
              'Relief', 'Pre', 'NDVI', 'Night', 'GPP',
              'ET', 'Erosion', 'Disturb', 'Water', 'Rail', 'Road',
              'Rock', 'DEM', 'Aspect']]
    y = data['Frozen2005']

    # Retain the original OID number
    if 'OID_' in data.columns:
        OID = data['OID_']
    else:
        raise ValueError("The 'OID_' column was not found in the original data!")

    # 3. Load model
    model_path = r'F:\SHAP\xgb_2020.json'
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(model_path)
    print("The model has been loaded successfully!")

    # 5. Analysis of SHAP interpretability
    output_dir = r'F:\SHAP\2020'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    explainer = shap.TreeExplainer(xgb_model)

    print("SHAP Calculating...")
    shap_values = []
    for i in tqdm(range(len(X))):
        shap_value = explainer.shap_values(X.iloc[i:i+1], approximate=True)
        shap_values.append(shap_value[0])

    # Convert the SHAP value to a DataFrame
    shap_values_df = pd.DataFrame(shap_values, columns=X.columns)
    shap_values_df['OID_'] = OID

    # Ensure the consistency of the index
    data = data.reset_index(drop=True)
    shap_values_df = shap_values_df.reset_index(drop=True)

    # Merge the SHAP values with the original data
    output_csv_path = os.path.join(output_dir, 'shap_2020.csv')
    merged_data = pd.concat([data, shap_values_df], axis=1)
    merged_data.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"SHAP values are saved in: {output_csv_path}")

if __name__ == '__main__':
    main()