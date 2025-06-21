This document outlines the steps taken to implement the methodologies described in the research paper. The following Python files are used sequentially in the analysis:

1.XGBoost Modeling: The first step involves building a predictive model using the XGBoost algorithm, which is implemented in the first Python file.

2.Prediction and Error Analysis: The second file uses the trained XGBoost model to make predictions and assess the prediction errors.

3.SHAP for Explainable Machine Learning: In the third step, we utilize SHAP to interpret the model's predictions, providing insights into feature importance.

4.Exploratory Clustering with XMeans: Before applying KMeans clustering, we conduct exploratory analysis using the XMeans clustering algorithm to determine the optimal number of clusters.

5.KMeans Clustering: The fifth file implements KMeans clustering on the dataset, segmenting the data into distinct clusters based on the features.

6.Causal Recursive Analysis with SAM Model: In the sixth step, we apply the Structural Additive Model (SAM) to perform causal recursive analysis, examining relationships between variables.

7.Threshold-Based Relationship Extraction: The seventh file extracts relationships based on predefined thresholds, allowing for focused analysis of significant interactions.

8.Jaccard Sensitivity Analysis: Finally, we conduct a Jaccard sensitivity analysis to assess the robustness of the thresholds chosen in the previous step.

Data
A dataset of 3000 sample records is provided for use in these analyses, which can be found in the project directory.
