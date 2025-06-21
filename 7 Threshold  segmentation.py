import os
import pickle
import networkx as nx
import numpy as np


def save_filtered_graphs(input_file, output_folder, thresholds):

    # Load the overall causal diagram
    with open(input_file, "rb") as f:
        causal_graph = pickle.load(f)
    os.makedirs(output_folder, exist_ok=True)

    # Traverse each threshold
    for t in thresholds:
        print(f"Processing t = {t:.1f}...")

        # Filter the edges based on the threshold
        filtered_edges = [
            (u, v, w) for (u, v, w) in causal_graph.edges(data=True)
            if abs(w.get('weight', 0)) >= t
        ]

        # Create the filtered graph
        filtered_graph = nx.DiGraph()
        filtered_graph.add_weighted_edges_from([(u, v, w['weight']) for u, v, w in filtered_edges])

        # Save the filtered causal diagram
        output_file = os.path.join(output_folder, f"SAM2_DISCOVERY_{t:.1f}.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(filtered_graph, f)
        print(f"The causal diagram is saved as: {output_file}")

    print("All causal diagrams have been successfully saved!")


def main():
    input_file = r"F:\Python file\Causation\SAM_DISCOVERY_0.6.pkl"
    output_folder = r"F:\Python file\Causation\second_layer_try"

    # Define the threshold range (from 0 to 1, every 0.1)
    thresholds = np.arange(0, 1.1, 0.1)
    save_filtered_graphs(input_file, output_folder, thresholds)

if __name__ == "__main__":
    main()