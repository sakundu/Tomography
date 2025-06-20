#!/usr/bin/env python3
"""
Phase 2 clustering and pruning script.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import igraph as ig
import leidenalg as la
from itertools import chain  # faster flattening of neighbor lists
import time  # for profiling

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
    edge_df['Source'] = edge_df['Source'].astype(str).str.replace(r'[{}\\]', '', regex=True)
    edge_df['Sink']   = edge_df['Sink'].astype(str).str.replace(r'[{}\\]', '', regex=True)
    src = node_df.rename(columns={'Instance':'Source','PT_X':'Source_X','PT_Y':'Source_Y'})
    snk = node_df.rename(columns={'Instance':'Sink','PT_X':'Sink_X','PT_Y':'Sink_Y'})
    edge_df = edge_df.merge(src[['Source','Source_X','Source_Y']], on='Source', how='left')
    edge_df = edge_df.merge(snk[['Sink','Sink_X','Sink_Y']], on='Sink', how='left')
    edge_df['Length'] = np.sqrt((edge_df['Source_X']-edge_df['Sink_X'])**2 + (edge_df['Source_Y']-edge_df['Sink_Y'])**2)
    return edge_df


def compute_raw_weights(edge_df, alpha):
    df = edge_df.dropna(subset=['Slack','ClockPeriod','Length']).copy()
    avg_len = df['Length'].mean()
    records = []
    for _, row in df.iterrows():
        L = row['Length']
        if L <= 0:
            continue
        slack = float(row['Slack'])
        cp = float(row['ClockPeriod'])
        if cp <= 0:
            continue
        weight = (avg_len/L) + alpha*(1 - slack/cp)
        records.append({'Source':row['Source'],'Sink':row['Sink'],'slack':slack,'weight':weight})
    return records


def build_graph_from_records(records):
    edges = [(r['Source'],r['Sink'],r['weight']) for r in records if r['weight']>0]
    G = ig.Graph.TupleList(edges, directed=True, edge_attrs=['weight'])
    return G


def get_seed_nodes(records):
    return set(r['Source'] for r in records if r['slack']<0) | set(r['Sink'] for r in records if r['slack']<0)


def get_khop_nodes(G, seeds, k):
    # map node names to indices once per call
    name_to_idx = {v['name']: i for i, v in enumerate(G.vs)}
    seed_indices = [name_to_idx[n] for n in seeds if n in name_to_idx]
    if not seed_indices:
        return set()
    # get k-hop neighborhoods and flatten efficiently
    neigh = G.neighborhood(seed_indices, order=k, mode='all')
    idxs = set(chain.from_iterable(neigh))
    return {G.vs[i]['name'] for i in idxs}


def run_leiden_on_edges(records):
    edges = [(r['Source'],r['Sink'],r['weight']) for r in records if r['weight']>0]
    G = ig.Graph.TupleList(edges, directed=True, edge_attrs=['weight'])
    partition = la.find_partition(G, la.RBConfigurationVertexPartition, weights='weight')
    clusters = {G.vs[node]['name']:cid for cid,nodes in enumerate(partition) for node in nodes}
    return clusters


def filter_clusters(clusters, min_size):
    df = pd.DataFrame(list(clusters.items()), columns=['Name','Cluster_id'])
    counts = df['Cluster_id'].value_counts()
    keep = counts[counts>=min_size].index
    return {name:cid for name,cid in clusters.items() if cid in set(keep)}


def stats_from_clusters(clusters):
    sizes = np.array(list(pd.Series(list(clusters.values())).value_counts().values))
    return {
        'num_clusters': len(sizes),
        'avg_size': sizes.mean() if len(sizes)>0 else 0,
        'median_size': np.median(sizes) if len(sizes)>0 else 0,
        'std_size': sizes.std() if len(sizes)>0 else 0,
        'min_size': sizes.min() if len(sizes)>0 else 0,
        'max_size': sizes.max() if len(sizes)>0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 2 Leiden clustering sweep')
    parser.add_argument('design')
    parser.add_argument('data_dir')
    parser.add_argument('--min_cluster_size', type=int, required=True)
    parser.add_argument('--k_values', nargs='+', type=int, required=True)
    parser.add_argument('--alpha_values', nargs='+', type=float, required=True)
    parser.add_argument('--ctypes', nargs='+', choices=['prune_then_cluster','cluster_then_prune'], required=True)
    parser.add_argument('--single_k2', action='store_true',
                        help='Run only for k=2 for faster profiling')
    args = parser.parse_args()

    node_df, edge_df = load_data(args.design, args.data_dir)
    edge_df = compute_length(edge_df, node_df)
    # Precompute raw weights once per alpha inside loop
    summary = []
    # if requested, restrict to k=2 only
    if args.single_k2:
        print("Single-k2 mode: restricting k_values to [2]")
        args.k_values = [2]
    for alpha in args.alpha_values:
        records = compute_raw_weights(edge_df, alpha)
        full_G = build_graph_from_records(records)
        seed_nodes = get_seed_nodes(records)
        for k in args.k_values:
            print(f"Processing k={k}...")
            Sk = seed_nodes.copy()
            if k > 0:
                t_k0 = time.perf_counter()
                Sk = get_khop_nodes(full_G, seed_nodes, k)
                t_k1 = time.perf_counter()
                print(f"  [Profile] k-hop selection time: {t_k1 - t_k0:.2f}s, selected nodes: {len(Sk)}")
            for ctype in args.ctypes:
                print(f"  Clustering mode: {ctype} (min_size={args.min_cluster_size})")
                # perform clustering and pruning with timing
                t_c0 = time.perf_counter()
                if ctype == 'prune_then_cluster':
                    # first prune graph to k-hop, then cluster, then prune small clusters
                    sub_rec = [r for r in records if r['Source'] in Sk and r['Sink'] in Sk]
                    clusters = run_leiden_on_edges(sub_rec)
                    clusters = filter_clusters(clusters, args.min_cluster_size)
                else:  # cluster_then_prune
                    # cluster full graph, then restrict to k-hop nodes, then prune small clusters
                    clusters_full = run_leiden_on_edges(records)
                    # restrict to k-hop set
                    clusters_restricted = {name:cid for name,cid in clusters_full.items() if name in Sk}
                    clusters = filter_clusters(clusters_restricted, args.min_cluster_size)
                t_c1 = time.perf_counter()
                # compute counts
                n_instances = len(clusters)
                n_clusters = len(set(clusters.values()))
                print(f"    [Profile] clustering+prune time: {t_c1 - t_c0:.2f}s, clusters: {n_clusters}, instances: {n_instances}")
                # save cluster CSV and DEF
                cluster_csv = f"{args.design}_k{k}_alpha{alpha}_ctype{ctype}_clusters.csv"
                pd.DataFrame(list(clusters.items()), columns=['Name','Cluster_id']).to_csv(cluster_csv, index=False)
                outfile = f"{args.design}_k{k}_alpha{alpha}_ctype{ctype}.def"
                gen_inst_group_def(cluster_csv=cluster_csv,
                                   cluster_def=os.path.abspath(outfile),
                                   design=args.design)
                print(f"    DEF written: {outfile}")
                # collect stats
                st = stats_from_clusters(clusters)
                st.update({'k': k, 'alpha': alpha, 'ctype': ctype})
                summary.append(st)
    # write summary table
    pd.DataFrame(summary).to_csv(f"{args.design}_phase2_summary.csv", index=False)
    print("Phase 2 summary written.")

if __name__=='__main__':
    main()
