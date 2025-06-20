#!/usr/bin/env python3
"""
Comprehensive synthesis cluster mapping script that generates clusters for multiple 
parameter combinations and maps them to synthesis netlist.

This script implements the full pipeline described in the updated README:
1. Generate clusters on post-route database for all parameter combinations
2. Map clusters to synthesis netlist for each combination
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
import argparse
from itertools import product
import time

# Add paths for importing the required modules
CLUSTERING_PATH = '/home/fetzfs_projects/NetlistTomography/sakundu/iccad24_tomography_testcases/ca53_gf12_diff_fps/clustering_a2'
MAPPING_PATH = '/home/fetzfs_projects/NetlistTomography/sakundu/iccad24_tomography_testcases/ca53_gf12_diff_fps/mapping_a2'
SCRATCH_PAD = '/home/fetzfs_projects/NetlistTomography/sakundu/scratch_pad'

if CLUSTERING_PATH not in sys.path:
    sys.path.append(CLUSTERING_PATH)
if MAPPING_PATH not in sys.path:
    sys.path.append(MAPPING_PATH)
if SCRATCH_PAD not in sys.path:
    sys.path.append(SCRATCH_PAD)

try:
    from run_cluster import load_data, compute_length, filter_and_weight, build_graph, run_leiden
    from run_phase2 import compute_raw_weights, build_graph_from_records, get_seed_nodes, get_khop_nodes, run_leiden_on_edges, filter_clusters
    from mapping_analysis import GraphAnalyzer
    from gen_cluster import gen_inst_group_def
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)


class ComprehensiveClusterMapper:
    def __init__(self, design_post_route_dir, design_post_synth_dir, design_name):
        self.design_post_route_dir = design_post_route_dir
        self.design_post_synth_dir = design_post_synth_dir
        self.design_name = design_name
        
        # Parameter combinations
        self.k_values = [0, 1, 2]
        self.min_size = 200
        self.ctype = "prune_then_cluster"
        self.alpha_values = [0.0, 0.5, 1.0, 2.0]
        
        # Data structures
        self.synthesis_nodes = set()
        self.post_route_nodes = set()
        self.matched_nodes = set()
        self.missing_nodes = set()
        
        # Cache for mapping analysis (shared across parameter combinations)
        self.mapping_analyzer = None
        self.nearest_mappings_cache = None
        
        # Results storage
        self.all_results = []
        
        print(f"Initialized mapper for design: {self.design_name}")
        print(f"Parameter combinations: {len(self.k_values) * len(self.alpha_values)} total")

    def load_base_data(self):
        """Load synthesis and post-route nodes for matching"""
        print("="*60)
        print("LOADING BASE DATA")
        print("="*60)
        
        # Load synthesis nodes
        synthesis_nodes_file = os.path.join(self.design_post_synth_dir, f"{self.design_name}_nodes.csv")
        print(f"Loading synthesis nodes from: {synthesis_nodes_file}")
        
        if not os.path.exists(synthesis_nodes_file):
            raise FileNotFoundError(f"Synthesis nodes file not found: {synthesis_nodes_file}")
        
        synthesis_df = pd.read_csv(synthesis_nodes_file)
        self.synthesis_nodes = set(synthesis_df['Instance'].astype(str))
        
        # Load post-route nodes
        post_route_nodes_file = os.path.join(self.design_post_route_dir, f"{self.design_name}_nodes.csv")
        print(f"Loading post-route nodes from: {post_route_nodes_file}")
        
        if not os.path.exists(post_route_nodes_file):
            raise FileNotFoundError(f"Post-route nodes file not found: {post_route_nodes_file}")
        
        post_route_df = pd.read_csv(post_route_nodes_file)
        self.post_route_nodes = set(post_route_df['Instance'].astype(str))
        
        # Find matched and missing nodes
        self.matched_nodes = self.synthesis_nodes.intersection(self.post_route_nodes)
        self.missing_nodes = self.synthesis_nodes - self.post_route_nodes
        
        print(f"Synthesis nodes: {len(self.synthesis_nodes):,}")
        print(f"Post-route nodes: {len(self.post_route_nodes):,}")
        print(f"Matched nodes: {len(self.matched_nodes):,}")
        print(f"Missing nodes: {len(self.missing_nodes):,}")
        
        # Initialize mapping analyzer for missing nodes
        if self.missing_nodes:
            print("Initializing mapping analyzer for missing nodes...")
            self.mapping_analyzer = GraphAnalyzer(self.design_post_synth_dir, self.design_post_route_dir, self.design_name)
            self.mapping_analyzer.load_data()
            self.mapping_analyzer.analyze_missing_instances()
            
            # Cache the nearest mappings
            if hasattr(self.mapping_analyzer, 'nearest_mappings') and self.mapping_analyzer.nearest_mappings:
                self.nearest_mappings_cache = {
                    mapping['missing_instance']: mapping['nearest_matched_instance']
                    for mapping in self.mapping_analyzer.nearest_mappings
                    if mapping['missing_instance'] in self.synthesis_nodes
                }
                print(f"Cached {len(self.nearest_mappings_cache):,} nearest mappings")
            else:
                self.nearest_mappings_cache = {}
                print("Warning: No nearest mappings found")

    def generate_single_clustering(self, k, alpha):
        """Generate clustering for a single parameter combination"""
        print(f"\nGenerating clustering for k={k}, alpha={alpha}")
        
        try:
            # Load and preprocess data
            node_df, edge_df = load_data(self.design_name, self.design_post_route_dir)
            edge_df = compute_length(edge_df, node_df)
            
            # Use run_phase2 approach for clustering
            records = compute_raw_weights(edge_df, alpha)
            full_G = build_graph_from_records(records)
            seed_nodes = get_seed_nodes(records)
            
            # Get k-hop nodes
            if k == 0:
                Sk = seed_nodes.copy()
            else:
                Sk = get_khop_nodes(full_G, seed_nodes, k)
            
            print(f"  Selected {len(Sk):,} nodes for k={k}")
            
            # Apply prune_then_cluster strategy
            sub_rec = [r for r in records if r['Source'] in Sk and r['Sink'] in Sk]
            clusters = run_leiden_on_edges(sub_rec)
            clusters = filter_clusters(clusters, self.min_size)
            
            print(f"  Generated {len(set(clusters.values())):,} clusters with {len(clusters):,} instances")
            
            # Add unassigned nodes with cluster ID = -1
            all_post_route_nodes = set(node_df['Instance'].astype(str))
            final_clusters = {}
            
            # Assign cluster IDs to clustered nodes
            for node, cluster_id in clusters.items():
                final_clusters[node] = cluster_id
            
            # Assign -1 to unassigned nodes
            for node in all_post_route_nodes:
                if node not in final_clusters:
                    final_clusters[node] = -1
            
            print(f"  Final mapping: {sum(1 for cid in final_clusters.values() if cid != -1):,} clustered, "
                  f"{sum(1 for cid in final_clusters.values() if cid == -1):,} unassigned")
            
            return final_clusters
            
        except Exception as e:
            print(f"Error generating clustering for k={k}, alpha={alpha}: {e}")
            return {}

    def map_to_synthesis(self, post_route_clusters, k, alpha):
        """Map post-route clusters to synthesis netlist"""
        print(f"  Mapping to synthesis netlist...")
        
        synthesis_clusters = {}
        matched_assigned = 0
        missing_mapped = 0
        missing_failed = 0
        
        # Assign clusters to matched nodes
        for node in self.matched_nodes:
            if node in post_route_clusters:
                synthesis_clusters[node] = post_route_clusters[node]
                matched_assigned += 1
        
        # Assign clusters to missing nodes using cached mappings
        for missing_node in self.missing_nodes:
            if missing_node in self.nearest_mappings_cache:
                nearest_node = self.nearest_mappings_cache[missing_node]
                if nearest_node and nearest_node in post_route_clusters:
                    synthesis_clusters[missing_node] = post_route_clusters[nearest_node]
                    missing_mapped += 1
                else:
                    synthesis_clusters[missing_node] = -1
                    missing_failed += 1
            else:
                synthesis_clusters[missing_node] = -1
                missing_failed += 1
        
        # Verify we have all synthesis nodes
        total_synthesis_assigned = len(synthesis_clusters)
        coverage = total_synthesis_assigned / len(self.synthesis_nodes) * 100
        
        print(f"    Matched nodes assigned: {matched_assigned:,}")
        print(f"    Missing nodes mapped: {missing_mapped:,}")
        print(f"    Missing nodes failed: {missing_failed:,}")
        print(f"    Total synthesis coverage: {coverage:.2f}%")
        
        return synthesis_clusters, {
            'matched_assigned': matched_assigned,
            'missing_mapped': missing_mapped,
            'missing_failed': missing_failed,
            'total_coverage': coverage
        }

    def generate_output_files(self, synthesis_clusters, k, alpha, stats):
        """Generate CSV and DEF files for a parameter combination"""
        # Create cluster DataFrame (excluding -1 assignments for DEF)
        cluster_data = []
        clustered_count = 0
        unassigned_count = 0
        
        for node, cluster_id in synthesis_clusters.items():
            if cluster_id != -1:
                cluster_data.append({'Name': node, 'Cluster_id': cluster_id})
                clustered_count += 1
            else:
                unassigned_count += 1
        
        if not cluster_data:
            print(f"    Warning: No clustered nodes for k={k}, alpha={alpha}")
            return
        
        # Save cluster CSV
        cluster_df = pd.DataFrame(cluster_data)
        cluster_csv = f"{self.design_name}_k{k}_alpha{alpha}_synthesis_clusters.csv"
        cluster_df.to_csv(cluster_csv, index=False)
        
        # Generate DEF file
        def_file = f"{self.design_name}_k{k}_alpha{alpha}_synthesis.def"
        try:
            gen_inst_group_def(cluster_csv, def_file, self.design_name)
            print(f"    Generated: {cluster_csv} ({clustered_count:,} nodes)")
            print(f"    Generated: {def_file}")
        except Exception as e:
            print(f"    Error generating DEF: {e}")
        
        # Compute cluster statistics
        if clustered_count > 0:
            cluster_sizes = cluster_df['Cluster_id'].value_counts()
            cluster_stats = {
                'num_clusters': len(cluster_sizes),
                'avg_cluster_size': cluster_sizes.mean(),
                'median_cluster_size': cluster_sizes.median(),
                'min_cluster_size': cluster_sizes.min(),
                'max_cluster_size': cluster_sizes.max(),
                'clustered_nodes': clustered_count,
                'unassigned_nodes': unassigned_count
            }
            
            print(f"    Cluster stats: {cluster_stats['num_clusters']} clusters, "
                  f"avg={cluster_stats['avg_cluster_size']:.1f}, "
                  f"max={cluster_stats['max_cluster_size']}")
        else:
            cluster_stats = {}
        
        return cluster_stats

    def run_complete_sweep(self, output_dir="."):
        """Run the complete parameter sweep"""
        print("="*60)
        print("STARTING COMPREHENSIVE CLUSTER MAPPING SWEEP")
        print("="*60)
        print(f"Parameters:")
        print(f"  k values: {self.k_values}")
        print(f"  alpha values: {self.alpha_values}")
        print(f"  min_size: {self.min_size}")
        print(f"  ctype: {self.ctype}")
        print(f"  Total combinations: {len(self.k_values) * len(self.alpha_values)}")
        
        # Load base data once
        self.load_base_data()
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Process each parameter combination
        total_combinations = len(self.k_values) * len(self.alpha_values)
        current_combination = 0
        
        for k, alpha in product(self.k_values, self.alpha_values):
            current_combination += 1
            print(f"\n{'='*60}")
            print(f"PROCESSING COMBINATION {current_combination}/{total_combinations}")
            print(f"k={k}, alpha={alpha}, min_size={self.min_size}, ctype={self.ctype}")
            print(f"{'='*60}")
            
            start_time = time.time()
            
            try:
                # Step 1: Generate post-route clustering
                post_route_clusters = self.generate_single_clustering(k, alpha)
                
                if not post_route_clusters:
                    print(f"  Skipping due to clustering failure")
                    continue
                
                # Step 2: Map to synthesis netlist
                synthesis_clusters, mapping_stats = self.map_to_synthesis(post_route_clusters, k, alpha)
                
                # Step 3: Generate output files
                cluster_stats = self.generate_output_files(synthesis_clusters, k, alpha, mapping_stats)
                
                # Record results
                result = {
                    'k': k,
                    'alpha': alpha,
                    'min_size': self.min_size,
                    'ctype': self.ctype,
                    'processing_time': time.time() - start_time,
                    **mapping_stats,
                    **(cluster_stats or {})
                }
                self.all_results.append(result)
                
                print(f"  Completed in {result['processing_time']:.1f}s")
                
            except Exception as e:
                print(f"  Error processing combination: {e}")
                continue
        
        # Generate summary report
        self.generate_summary_report(output_dir)
        
        print(f"\n{'='*60}")
        print("SWEEP COMPLETED")
        print(f"{'='*60}")
        print(f"Successfully processed {len(self.all_results)}/{total_combinations} combinations")

    def generate_summary_report(self, output_dir):
        """Generate comprehensive summary report"""
        if not self.all_results:
            print("No results to summarize")
            return
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(self.all_results)
        summary_file = os.path.join(output_dir, f"{self.design_name}_comprehensive_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\nSUMMARY REPORT:")
        print(f"  Saved: {summary_file}")
        
        # Print key statistics
        if len(summary_df) > 0:
            print(f"  Average coverage: {summary_df['total_coverage'].mean():.2f}%")
            print(f"  Best coverage: {summary_df['total_coverage'].max():.2f}%")
            print(f"  Average processing time: {summary_df['processing_time'].mean():.1f}s")
            
            # Best performing combination
            best_idx = summary_df['total_coverage'].idxmax()
            best_combo = summary_df.iloc[best_idx]
            print(f"  Best combination: k={best_combo['k']}, alpha={best_combo['alpha']} "
                  f"({best_combo['total_coverage']:.2f}% coverage)")

def main():
    # Default paths as specified in README
    design_post_route = "/home/fetzfs_projects/NetlistTomography/sakundu/iccad24_tomography_testcases/ca53_gf12_run2/run2_syn_cp_450_util_0.70/design_post_route"
    design_post_synth = "/home/fetzfs_projects/NetlistTomography/sakundu/iccad24_tomography_testcases/ca53_gf12_run2/run2_syn_cp_450_util_0.70/design_post_synth"
    design = "ca53_cpu"
    
    parser = argparse.ArgumentParser(description='Comprehensive synthesis clustering sweep')
    parser.add_argument('--design_post_route', default=design_post_route, 
                        help='Path to post-route design directory')
    parser.add_argument('--design_post_synth', default=design_post_synth, 
                        help='Path to post-synthesis design directory')
    parser.add_argument('--design', default=design, help='Design name')
    parser.add_argument('--output_dir', default='.', help='Output directory for generated files')
    parser.add_argument('--test_mode', action='store_true', 
                        help='Run only k=0, alpha=1.0 for testing')
    
    args = parser.parse_args()
    
    # Verify directories exist
    if not os.path.exists(args.design_post_route):
        print(f"Error: Post-route directory does not exist: {args.design_post_route}")
        sys.exit(1)
    
    if not os.path.exists(args.design_post_synth):
        print(f"Error: Post-synthesis directory does not exist: {args.design_post_synth}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Initialize mapper
    mapper = ComprehensiveClusterMapper(args.design_post_route, args.design_post_synth, args.design)
    
    # Optionally restrict parameters for testing
    if args.test_mode:
        print("TEST MODE: Running single combination (k=0, alpha=1.0)")
        mapper.k_values = [0]
        mapper.alpha_values = [1.0]
    
    # Run the complete sweep
    mapper.run_complete_sweep(args.output_dir)

if __name__ == "__main__":
    main()
