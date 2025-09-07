import pandas as pd
import networkx as nx
import copy as cp


class Config():
    def __init__(self, net_df, demand_df, all_od_paths):
        self.net_df = net_df
        self.demand_df = demand_df
        self.all_od_paths = all_od_paths


# ------------- get k shortest paths -------------
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


def load_OW_net_config(od_demand_file='OW_trips.csv', k=5):
    #k = args.k  # k shortest path
    OW_net = pd.read_csv('OW_net/OW_net.csv', sep=',')
    OW_net_df = OW_net.copy()
    OW_net_df['edge_flow'] = 0
    OW_net_df['edge_tt'] = 0
    # OW_df['cost'] = OW_df['free_flow_time']
    OW_net_df['weight'] = OW_net_df['free_flow_time']
    OW_net_df['_'] = '_'
    OW_net_df['od'] = OW_net_df["init_node"].astype(str) + OW_net_df['_'] + OW_net_df["term_node"].astype(str)
    OW_net_df = OW_net_df.drop(['_'], axis=1)
    #OW_net_df

    G = nx.from_pandas_edgelist(OW_net_df, 'init_node', 'term_node', ['free_flow_time', 'edge_flow', 'edge_tt', 'weight'], create_using=nx.DiGraph())
    node_xy = pd.read_excel('OW_net/OW_Node_XY.xlsx')  # X,Y position for good visualization
    # for better looking graph
    # pos_xy=dict([(i,(a,b)) for i, a,b in zip(node_xy.Node, node_xy.X,node_xy.Y)])
    pos_xy = dict([(i, (a, b)) for i, a, b in zip(node_xy.Node_num, node_xy.X, node_xy.Y)])

    for n, p in pos_xy.items():
        G.nodes[n]['pos_xy'] = p
    # demand
    # OW_demand_df = pd.read_excel('OW_Trips_and_XY.xlsx')  # old daa
    OW_demand_df = pd.read_csv(f'OW_net/{od_demand_file}', sep=',')
    OW_demand_df['_'] = '_'
    OW_demand_df['od'] = OW_demand_df["init_node"].astype(str) + OW_demand_df['_'] + OW_demand_df["term_node"].astype(str)
    del OW_demand_df['_']
    #print('OW_demand_df:', OW_demand_df)
    #OW_demand_df

    ksp_dict = {}
    for od in OW_demand_df.od.values.tolist():
        #print(od)
        o, d = int(od.split('_')[0]), int(od.split('_')[1])
        paths, lengths = k_shortest_paths(G=G, source=o, target=d, k=k, weight='weight')
        ksp_dict[od] = paths

    ow_config = Config(net_df=OW_net_df, demand_df=OW_demand_df, all_od_paths=ksp_dict)
    return ow_config



if __name__ == '__main__':
    args = load_OW_net_config()
    print(args.demand_df)
    print('args.all_od_paths=', args.all_od_paths)