import os
import geopandas as gpd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


input_file = r'D:\Fishnet1km\fishnet1km_connected2020.shp'
output_folder = r'D:\integrated data\Kmeans\explore2020'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cluster_fields = [
    'SHAP_CLC', 'SHAP_Win', 'SHAP_Tem', 'SHAP_Sun',
    'SHAP_San', 'SHAP_Cla', 'SHAP_Sil', 'SHAP_Slo', 'SHAP_Rel',
    'SHAP_Pre', 'SHAP_NDV', 'SHAP_Nig', 'SHAP_GPP',
    'SHAP_ET.', 'SHAP_Ero', 'SHAP_Dis', 'SHAP_Wat',
    'SHAP_Rai', 'SHAP_Roa', 'SHAP_Roc', 'SHAP_DEM',
    'SHAP_Asp', 'GEO_SHAP'
]

gdf = gpd.read_file(input_file)

missing_fields = [field for field in cluster_fields if field not in gdf.columns]
if missing_fields:
    raise ValueError(f"以下字段在矢量文件中不存在: {missing_fields}")

data = gdf[cluster_fields].replace(-9999, np.nan).dropna().values  # 删除缺失值
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Cluster quantity setting (according to the result of Xmeans)
n_clusters_list = [9]

# Perform cyclic clustering and save the results
for n_clusters in n_clusters_list:
    print(f"K-means clustering is underway, and the number of clusters is: {n_clusters}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_scaled)

    gdf_clustered = gdf.dropna(subset=cluster_fields).copy()
    gdf_clustered['Cluster_ID'] = cluster_labels

    output_file = os.path.join(output_folder, f'clustered2020_{n_clusters}clusters.shp')

    gdf_clustered.to_file(output_file)
    print(f"K-means of（{n_clusters} types）is saved in: {output_file}")

print("Mission completed！")