import os
import geopandas as gpd
import numpy as np
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.preprocessing import StandardScaler

# Enter the path of the vector file
input_file = r'D:\Fishnet1km\fishnet1km_label_connected.shp'
output_folder = r'D:\integrated data\Xmeans'
output_file = os.path.join(output_folder, 'fishnet1km_label_connected_clustered.shp')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# The fields used for clustering
cluster_fields = [
    'SHAP_r2000.1', 'SHAP_Wind.1', 'SHAP_Temerature.1', 'SHAP_Sunlight.1',
    'SHAP_Sand.1', 'SHAP_Clay.1', 'SHAP_Silt.1', 'SHAP_Slope.1', 'SHAP_Relief.1',
    'SHAP_Precipitat.1', 'SHAP_NDVI.1', 'SHAP_Nightlight.1', 'SHAP_GPP.1',
    'SHAP_ET.1', 'SHAP_Erosion.1', 'SHAP_Disturb.1', 'SHAP_Dis_Water.1',
    'SHAP_Dis_Railwa.1', 'SHAP_Dis_road.1', 'SHAP_RockDepth.1', 'SHAP_DEM.1',
    'SHAP_Aspect.1', 'Res2000', 'GEO_SHAP'
]

gdf = gpd.read_file(input_file)

missing_fields = [field for field in cluster_fields if field not in gdf.columns]
if missing_fields:
    raise ValueError(f"The following fields do not exist in the vector file: {missing_fields}")

# Extract the clustering data and standardize
data = gdf[cluster_fields].replace(-9999, np.nan).dropna().values
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Initialize the X-means algorithm
initial_centers = kmeans_plusplus_initializer(data_scaled, 2).initialize()
xmeans_instance = xmeans(data_scaled, initial_centers)

# Run X-means clustering
xmeans_instance.process()
clusters = xmeans_instance.get_clusters()
cluster_labels = np.zeros(len(data_scaled), dtype=int) - 1
for cluster_id, points in enumerate(clusters):
    cluster_labels[points] = cluster_id

# Add the clustering results back to the GeoDataFrame
gdf = gdf.dropna(subset=cluster_fields).copy()
gdf['Cluster_ID'] = cluster_labels

# Save the result to a new vector file
gdf.to_file(output_file)

print(f"The clustering results have been saved in: {output_file}")