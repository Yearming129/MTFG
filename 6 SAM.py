import geopandas as gpd
import pandas as pd
from cdt.causality.graph import SAM
import networkx as nx
import matplotlib.pyplot as plt
import pickle

def main():
    # Step 1: Load the shapefile data and extract the required fields
    shapefile_path = r'C:\Python file\Fishnet1km\fishnet1km_connected.shp'

    gdf = gpd.read_file(shapefile_path)

    selected_fields = [
        'r2000', 'long', 'lat', 'Wind', 'Temerature', 'Sunlight', 'Sand',
        'Clay', 'Silt', 'Slope', 'Relief', 'Precipitat', 'NDVI', 'Nightlight', 'GPP', 'ET', 'Erosion', 'Disturb',
        'Dis_Water', 'Dis_Railwa', 'Dis_road', 'RockDepth', 'DEM', 'Aspect', 'Frozen2000'
    ]

    for field in selected_fields:
        if field not in gdf.columns:
            raise ValueError(f"'{field}' does not exist in the shapefile.")

    data = gdf[selected_fields]

    # Step 2: Data preprocessing
    # Delete the rows containing missing values to avoid causal discovery problems caused by missing values
    data = data.dropna()

    # Convert the data to the Pandas DataFrame type (if it is not yet a DataFrame)
    data = pd.DataFrame(data)

    # Randomly select 3,000 pieces of data for analysis
    sample_size = 3000
    if len(data) > sample_size:
        print(f"The total amount of data is {len(data)} items, and {sample_size} items of data are randomly selected for analysis...")
        data = data.sample(n=sample_size, random_state=42)
    else:
        print(f"The total amount of data is {len(data)} items (less than {sample_size} items), and all the data is directly used for analysis.")

    # Step 3: Use SAM for causal discovery
    sam = SAM()
    causal_graph = sam.predict(data)

    # Save the causal diagram model to a file
    with open("causal_graph_model.pkl", "wb") as f:
        pickle.dump(causal_graph, f)
    print("The causal graph model has been saved as the 'causal_graph_model.pkl' file!")

#     # Optional: Redraw using the saved model
#     visualize_graph("causal_graph_model.pkl")
# def visualize_graph(model_path):
#     with open(model_path, "rb") as f:
#         causal_graph = pickle.load(f)

    # Step 4: Apply a threshold to filter weak causal relationships
    threshold = 0.6  # Define the threshold for edge weights
    weighted_edges = nx.get_edge_attributes(causal_graph, 'weight')
    filtered_edges = [(u, v, w) for (u, v, w) in causal_graph.edges(data=True) if abs(w.get('weight', 0)) >= threshold]
    filtered_graph = nx.DiGraph()
    filtered_graph.add_weighted_edges_from([(u, v, w['weight']) for u, v, w in filtered_edges])

    # Step 5: Visualize the causal diagram
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(causal_graph)
    nx.draw(
        causal_graph, pos, with_labels=True, node_size=3000, node_color='lightblue',
        font_size=10, font_weight='bold', edge_color='gray'
    )
    nx.draw_networkx_edge_labels(
        filtered_graph, pos, edge_labels={(u, v): f"{w['weight']:.2f}" for u, v, w in filtered_edges}

    plt.title("SAM Causal Diagram")
    plt.show()

    # Step 6: Analyze the results
    # Print the adjacency matrix of the causal graph
    adj_matrix = nx.adjacency_matrix(causal_graph).todense()
    print("The adjacency matrix of the causal graph:\n", adj_matrix)

    # Save the causal relationship as an edge list
    edges = list(causal_graph.edges())
    print("推断的因果关系:", edges)


if __name__ == "__main__":
    main()