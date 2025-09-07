# Complete script for k-shortest paths analysis with customizable k value
import pandas as pd
import networkx as nx
import numpy as np
import copy as cp
import os
import seaborn as sns
import matplotlib.pyplot as plt

# ------------- Get k shortest paths -------------
def k_shortest_paths(G, source, target, k=1, weight='weight'):
    # https://github.com/Mokerpoker/k_shortest_paths
    # G is a networkx graph.
    # source and target are the labels for the source and target of the path.
    # k is the amount of desired paths.
    # weight = 'weight' assumes a weighed graph. If this is undesired, use weight = None.

    A = [nx.dijkstra_path(G, source, target, weight=weight)]
    A_len = [sum([G[A[0][l]][A[0][l + 1]][weight] for l in range(len(A[0]) - 1)])]
    B = []

    for i in range(1, k):
        for j in range(0, len(A[-1]) - 1):
            Gcopy = cp.deepcopy(G)
            spurnode = A[-1][j]
            rootpath = A[-1][:j + 1]
            for path in A:
                if rootpath == path[0:j + 1]:  # and len(path) > j?
                    if Gcopy.has_edge(path[j], path[j + 1]):
                        Gcopy.remove_edge(path[j], path[j + 1])
                    if Gcopy.has_edge(path[j + 1], path[j]):
                        Gcopy.remove_edge(path[j + 1], path[j])
            for n in rootpath:
                if n != spurnode:
                    Gcopy.remove_node(n)
            try:
                spurpath = nx.dijkstra_path(Gcopy, spurnode, target, weight=weight)
                totalpath = rootpath + spurpath[1:]
                if totalpath not in B:
                    B += [totalpath]
            except nx.NetworkXNoPath:
                continue
        if len(B) == 0:
            break
        lenB = [sum([G[path[l]][path[l + 1]][weight] for l in range(len(path) - 1)]) for path in B]
        B = [p for _, p in sorted(zip(lenB, B))]
        A.append(B[0])
        A_len.append(sorted(lenB)[0])
        B.remove(B[0])

    return A, A_len

def generate_path_similarity_heatmap(paths):
    """
    Generate path similarity matrix and heatmap.
    
    Args:
        paths: List of paths, each path is a list of nodes
    
    Returns:
        similarity_matrix: Similarity matrix
    """
    n_paths = len(paths)
    similarity_matrix = np.zeros((n_paths, n_paths))
    
    # Calculate similarity between each pair of paths
    for i in range(n_paths):
        for j in range(n_paths):
            if i == j:
                similarity_matrix[i,j] = 1.0  # Self-similarity is 1
            else:
                # Convert paths to edge sets
                edges_i = set(zip(paths[i][:-1], paths[i][1:]))
                edges_j = set(zip(paths[j][:-1], paths[j][1:]))
                
                # Calculate Jaccard similarity
                inter = len(edges_i & edges_j)
                union = len(edges_i | edges_j)
                similarity_matrix[i,j] = inter / union if union > 0 else 0
                
    return similarity_matrix

def analyze_path_variance_with_heatmap(net_path, trips_path, k=15, output_dir='heatmaps'):
    """
    Analyze path variance and generate heatmaps.
    
    Args:
        net_path: Network file path
        trips_path: Trips file path
        k: Number of shortest paths
        output_dir: Heatmap output directory
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    df_net = pd.read_csv(net_path)
    df_trips = pd.read_csv(trips_path)
    
    # Build graph
    G = nx.DiGraph()
    for _, row in df_net.iterrows():
        G.add_edge(row['init_node'], row['term_node'], weight=row['free_flow_time'])
    
    # Extract OD pairs
    od_pairs = list(df_trips[['init_node', 'term_node']].drop_duplicates().itertuples(index=False, name=None))
    
    # Store heatmap data for all OD pairs
    heatmap_data = {}
    
    # Analyze each OD pair
    results = []
    for source, target in od_pairs:
        try:
            paths, costs = k_shortest_paths(G, source, target, k=k, weight='weight')
            
            # Generate similarity matrix
            similarity_matrix = generate_path_similarity_heatmap(paths)
            
            # Store heatmap data
            od_key = f"{source}→{target}"
            heatmap_data[od_key] = similarity_matrix
            
            # Generate and save heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(similarity_matrix, 
                       annot=True, 
                       cmap='YlOrRd',
                       vmin=0, 
                       vmax=1,
                       fmt='.2f')
            # plt.title(f'Path Similarity Heatmap for {od_key}')
            plt.savefig(os.path.join(output_dir, f'heatmap_{source}_{target}.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Calculate statistical metrics
            base_edges = set(zip(paths[0][:-1], paths[0][1:]))
            sims = []
            for path in paths[1:]:
                other_edges = set(zip(path, path[1:]))
                inter = len(base_edges & other_edges)
                union = len(base_edges | other_edges)
                sims.append(inter / union if union > 0 else 0)
            
            sim_mean = np.mean(sims)
            sim_var = np.var(sims)
            ff_mean = np.mean(costs)
            ff_var = np.var(costs)
            
            results.append({
                'OD': od_key,
                'MeanSimilarity': sim_mean,
                'SimilarityVariance': sim_var,
                'MeanFreeFlowTime': ff_mean,
                'VarianceFreeFlowTime': ff_var
            })
            
        except nx.NetworkXNoPath:
            results.append({
                'OD': f"{source}→{target}",
                'MeanSimilarity': None,
                'SimilarityVariance': None,
                'MeanFreeFlowTime': None,
                'VarianceFreeFlowTime': None
            })
    
    # Save heatmap data
    for od_key, matrix in heatmap_data.items():
        # Convert similarity matrix to DataFrame
        df_matrix = pd.DataFrame(matrix)
        # Add row and column labels
        df_matrix.index = [f'Path_{i+1}' for i in range(len(matrix))]
        df_matrix.columns = [f'Path_{i+1}' for i in range(len(matrix))]
        # Save as CSV
        df_matrix.to_csv(os.path.join(output_dir, f'heatmap_data_{od_key}.csv'))
    
    return pd.DataFrame(results), heatmap_data

# Run analysis and generate heatmaps
df_stats, heatmap_data = analyze_path_variance_with_heatmap("OW_net.csv", "OW_trips.csv", k=15)

# df_stats.to_csv('similarity.csv')

# View heatmap data for specific OD pair
od_key = "1→2"  # Example OD pair
if od_key in heatmap_data:
    print(f"Similarity matrix shape: {heatmap_data[od_key].shape}")
    print(heatmap_data[od_key])