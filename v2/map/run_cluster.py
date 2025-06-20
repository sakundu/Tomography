#!/usr/bin/env python3
"""
Script to perform Leiden clustering on a design's netlist based on edge slack and geometrical length.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import igraph as ig
import leidenalg as la

# Include scratch_pad path to import gen_inst_group_def
SCRATCH_PAD = '/home/fetzfs_projects/NetlistTomography/sakundu/scratch_pad'
if SCRATCH_PAD not in sys.path:
    sys.path.append(SCRATCH_PAD)
try:
    from gen_cluster import gen_inst_group_def  # type: ignore
except ImportError:
    print(f"Error: Cannot import gen_inst_group_def from {SCRATCH_PAD}/gen_cluster.py")
    sys.exit(1)


def load_data(design, data_dir):
    node_csv = os.path.join(data_dir, f"{design}_nodes.csv")
    edge_csv = os.path.join(data_dir, f"{design}_edges.csv")
    node_df = pd.read_csv(node_csv)
    edge_df = pd.read_csv(edge_csv, na_values=['INFINITE'])
    return node_df, edge_df


def compute_length(edge_df, node_df):
    # Clean node names
    edge_df['Source'] = edge_df['Source'].astype(str).str.replace(r'[{}\\]', '', regex=True)
    edge_df['Sink']   = edge_df['Sink'].astype(str).str.replace(r'[{}\\]', '', regex=True)

    # Merge source coordinates
    src = node_df.rename(columns={ 'Instance':'Source', 'PT_X':'Source_X', 'PT_Y':'Source_Y' })
    edge_df = edge_df.merge(src[['Source','Source_X','Source_Y']], on='Source', how='left')
    # Merge sink coordinates
    snk = node_df.rename(columns={ 'Instance':'Sink', 'PT_X':'Sink_X', 'PT_Y':'Sink_Y' })
    edge_df = edge_df.merge(snk[['Sink','Sink_X','Sink_Y']], on='Sink', how='left')

    # Compute Euclidean length
    edge_df['Length'] = np.sqrt((edge_df['Source_X'] - edge_df['Sink_X'])**2 +
                                 (edge_df['Source_Y'] - edge_df['Sink_Y'])**2)
    return edge_df


def filter_and_weight(edge_df, alpha=1.0):
    # Keep edges with numeric Slack and ClockPeriod
    df = edge_df.dropna(subset=['Slack','ClockPeriod','Length']).copy()
    # Compute average length
    avg_len = df['Length'].mean()
    weighted = []
    for _, row in df.iterrows():
        L = row['Length']
        # skip zero-length edges to avoid division by zero
        if L <= 0:
            continue
        slack = float(row['Slack'])
        cp = float(row['ClockPeriod'])
        # skip zero or negative clock periods
        if cp <= 0:
            continue
        # Skip if length > avg and slack >= 0
        if L > avg_len and slack >= 0:
            continue
        weight = (avg_len / L) + alpha * (1 - slack / cp)
        # skip non-positive weights to satisfy Leiden
        if weight <= 0:
            continue
        weighted.append((row['Source'], row['Sink'], weight))
    return weighted


def build_graph(weighted_edges):
    # Create directed igraph with edge attribute 'weight'
    G = ig.Graph.TupleList(weighted_edges, directed=True, edge_attrs=['weight'])
    return G


def run_leiden(G):
    partition = la.find_partition(G, la.RBConfigurationVertexPartition, weights='weight')
    clusters = {}
    for cid, nodes in enumerate(partition):
        for node in nodes:
            clusters[G.vs[node]['name']] = cid
    return clusters


def write_stats_and_def(clusters, design, alpha):
    # clusters: dict name->cid
    df = pd.DataFrame(list(clusters.items()), columns=['Name','Cluster_id'])
    # prune small clusters if min_cluster_size provided in global args
    min_size = globals().get('_MIN_CLUSTER_SIZE_', 1)
    counts = df['Cluster_id'].value_counts()
    keep = counts[counts >= min_size].index.tolist()
    df = df[df['Cluster_id'].isin(keep)]

    # Save cluster CSV for DEF generator
    cluster_csv = f"{design}_clusters.csv"
    df.to_csv(cluster_csv, index=False)
    # Generate DEF file in current directory to ensure path exists
    def_file = os.path.abspath(f"{design}.def")
    gen_inst_group_def(cluster_csv, def_file, design)
    # Compute cluster sizes
    sizes = df['Cluster_id'].value_counts().values
    print(f"Number of clusters: {len(sizes)}")
    print(f"Average cluster size: {sizes.mean():.2f}")
    print(f"Median cluster size: {np.median(sizes):.2f}")
    print(f"Std dev cluster size: {sizes.std():.2f}")
    print(f"Min cluster size: {sizes.min()}")
    print(f"Max cluster size: {sizes.max()}")


def main():
    parser = argparse.ArgumentParser(description='Leiden clustering based on edge slack and geometry')
    parser.add_argument('design', help='Design name (prefix of CSV files)')
    parser.add_argument('data_dir', help='Directory containing node and edge CSVs')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weighting parameter alpha')
    parser.add_argument('--min_cluster_size', type=int, default=1,
                        help='Minimum cluster size to include in output DEF')
    args = parser.parse_args()
    # set global for write_stats_and_def
    globals()['_MIN_CLUSTER_SIZE_'] = args.min_cluster_size

    node_df, edge_df = load_data(args.design, args.data_dir)
    print("Loaded data")
    edge_df = compute_length(edge_df, node_df)
    print("Computed edge lengths")
    weighted_edges = filter_and_weight(edge_df, alpha=args.alpha)
    print(f"Filtered and weighted edges: {len(weighted_edges)}")
    G = build_graph(weighted_edges)
    print("Built igraph graph")
    clusters = run_leiden(G)
    print(f"Found clusters: {len(set(clusters.values()))}")
    write_stats_and_def(clusters, args.design, args.alpha)
    print("DEF and statistics written")


if __name__ == '__main__':
    main()
