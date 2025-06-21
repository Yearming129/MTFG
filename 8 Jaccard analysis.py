import matplotlib.pyplot as plt
import os
import pickle

def calculate_jaccard_index(graph1, graph2):

    edges1 = set(graph1.edges())
    edges2 = set(graph2.edges())
    intersection = len(edges1 & edges2)
    union = len(edges1 | edges2)
    return intersection / union if union != 0 else 0

def main():
    # The path of the causal diagram folder
    folder_path = r'F:\Python file\Causation\second_layer_try'

    # Traverse all the causal diagram files in the folder (sorted by threshold)
    pkl_files = sorted(
        [f for f in os.listdir(folder_path) if f.startswith("SAM2_DISCOVERY_") and f.endswith(".pkl")],
        key=lambda x: float(x.split("_")[-1][:-4])
    )
    thresholds = [float(f.split("_")[-1][:-4]) for f in pkl_files]

    # Save Jaccard Index
    jaccard_indices = []

    # Traverse each file
    previous_graph = None
    for pkl_file in pkl_files:
        file_path = os.path.join(folder_path, pkl_file)
        print(f"Load the causal diagram file: {file_path}")

        with open(file_path, "rb") as f:
            causal_graph = pickle.load(f)

        # Calculating the Jaccard Index
        if previous_graph is not None:
            jaccard_index = calculate_jaccard_index(previous_graph, causal_graph)
            jaccard_indices.append(jaccard_index)
        else:
            jaccard_indices.append(None)  # 第一个图没有 Jaccard Index

        # Update the previous graph
        previous_graph = causal_graph

    # Draw the stability chart of the Jaccard Index
    plt.figure(figsize=(6, 6))
    thresholds_for_jaccard = thresholds[1:]
    plt.plot(thresholds_for_jaccard, jaccard_indices[1:], marker='o', label='Jaccard Index')
    plt.xlabel('Threshold (t)')
    plt.ylabel('Jaccard Index')
    plt.title('Jaccard Index Stability Across Thresholds')
    plt.grid(True)
    plt.legend()
    plt.savefig("jaccard_stability.jpg")
    plt.show()

if __name__ == "__main__":
    main()