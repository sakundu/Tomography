#!/usr/bin/env python3
"""
Multi-source BFS mapping analysis script.
Analyzes two designs to find hop distances between missing and matched instances.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, deque
import argparse
import os
import sys
import time

class GraphAnalyzer:
    def __init__(self, dir1, dir2, design):
        self.dir1 = dir1
        self.dir2 = dir2
        self.design = design
        
        # Data structures for analysis
        self.dir1_nodes = {}  # instance -> {cell, slack, ...}
        self.dir2_nodes = {}  # instance -> {cell, slack, ...}
        
        self.forward_graph = defaultdict(list)  # source -> [sinks]
        self.reverse_graph = defaultdict(list)  # sink -> [sources]
        
        self.matched_instances = set()  # instances present in both
        self.missing_instances = set()  # instances in dir1 but not in dir2
        
        # Results storage
        self.results = []
        self.nearest_mappings = []  # Store nearest mapping between forward/reverse for each missing instance
        
        # Cache for memoization of BFS results
        self.forward_cache = {}  # node -> (matched_node, distance)
        self.reverse_cache = {}  # node -> (matched_node, distance)
        
    def load_data(self):
        """Load nodes and edges from both directories with optimizations."""
        print("Loading data...")
        
        # Load nodes from both directories
        nodes1_file = os.path.join(self.dir1, f"{self.design}_nodes.csv")
        nodes2_file = os.path.join(self.dir2, f"{self.design}_nodes.csv")
        
        print(f"Loading nodes from {nodes1_file}")
        start_time = time.time()
        df1_nodes = pd.read_csv(nodes1_file)
        print(f"Reading CSV took: {time.time() - start_time:.2f} seconds")
        
        print(f"Loading nodes from {nodes2_file}")
        start_time = time.time()
        df2_nodes = pd.read_csv(nodes2_file)
        print(f"Reading CSV took: {time.time() - start_time:.2f} seconds")
        
        # Optimized node loading using pandas native methods
        print("Converting to dictionaries...")
        start_time = time.time()
        
        # Fast approach using zip for vectorized operations
        instances1 = df1_nodes['Instance'].values
        cells1 = df1_nodes['Cell'].values
        slacks1 = df1_nodes['Slack'].values
        clock_periods1 = df1_nodes['ClockPeriod'].values
        
        self.dir1_nodes = {
            inst: {'cell': cell, 'slack': slack, 'clock_period': cp}
            for inst, cell, slack, cp in zip(instances1, cells1, slacks1, clock_periods1)
        }
        print(f"Dir1 nodes conversion took: {time.time() - start_time:.2f} seconds")
        
        start_time = time.time()
        instances2 = df2_nodes['Instance'].values
        cells2 = df2_nodes['Cell'].values
        slacks2 = df2_nodes['Slack'].values
        clock_periods2 = df2_nodes['ClockPeriod'].values
        
        self.dir2_nodes = {
            inst: {'cell': cell, 'slack': slack, 'clock_period': cp}
            for inst, cell, slack, cp in zip(instances2, cells2, slacks2, clock_periods2)
        }
        print(f"Dir2 nodes conversion took: {time.time() - start_time:.2f} seconds")
        
        # Load edges from dir1 to build the graph
        edges1_file = os.path.join(self.dir1, f"{self.design}_edges.csv")
        print(f"Loading edges from {edges1_file}")
        start_time = time.time()
        df1_edges = pd.read_csv(edges1_file)
        print(f"Reading edges CSV took: {time.time() - start_time:.2f} seconds")
        
        # Optimized graph building using vectorized operations
        print("Building graphs...")
        start_time = time.time()
        
        # Use pandas vectorized operations for faster graph building
        sources = df1_edges['Source'].values
        sinks = df1_edges['Sink'].values
        
        for source, sink in zip(sources, sinks):
            self.forward_graph[source].append(sink)
            self.reverse_graph[sink].append(source)
        print(f"Building graphs took: {time.time() - start_time:.2f} seconds")
        
        # Identify matched and missing instances
        print("Computing instance sets...")
        start_time = time.time()
        dir1_instances = set(self.dir1_nodes.keys())
        dir2_instances = set(self.dir2_nodes.keys())
        
        self.matched_instances = dir1_instances.intersection(dir2_instances)
        self.missing_instances = dir1_instances - dir2_instances
        print(f"Set operations took: {time.time() - start_time:.2f} seconds")
        
        print(f"Dir1 instances: {len(dir1_instances)}")
        print(f"Dir2 instances: {len(dir2_instances)}")
        print(f"Matched instances: {len(self.matched_instances)}")
        print(f"Missing instances: {len(self.missing_instances)}")
        
    def bfs_find_nearest_matched(self, start_instance, graph, cache):
        """
        Optimized BFS to find nearest matched instance with memoization.
        Returns (nearest_matched_instance, distance)
        """
        # Check cache first
        if start_instance in cache:
            return cache[start_instance]
            
        if start_instance in self.matched_instances:
            result = (start_instance, 0)
            cache[start_instance] = result
            return result
            
        # Use deque for O(1) append/popleft operations
        queue = deque([(start_instance, 0)])
        visited = {start_instance}
        path_nodes = []  # Track nodes in current path for caching
        
        while queue:
            current, distance = queue.popleft()
            path_nodes.append((current, distance))
            
            # Get neighbors efficiently - use .get() to avoid KeyError
            neighbors = graph.get(current, [])
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                    
                visited.add(neighbor)
                
                # Check if neighbor result is already cached
                if neighbor in cache:
                    matched_node, cached_dist = cache[neighbor]
                    if matched_node is not None:  # Valid cached result
                        result = (matched_node, distance + 1 + cached_dist)
                        
                        # Cache all nodes in the path
                        for path_node, path_dist in path_nodes:
                            if path_node not in cache:  # Avoid overwriting
                                cache[path_node] = (matched_node, result[1] - path_dist)
                        
                        return result
                
                # If we found a matched instance, return it
                if neighbor in self.matched_instances:
                    result = (neighbor, distance + 1)
                    
                    # Cache all nodes in the path
                    for path_node, path_dist in path_nodes:
                        if path_node not in cache:  # Avoid overwriting
                            cache[path_node] = (neighbor, result[1] - path_dist)
                    cache[neighbor] = (neighbor, 0)
                    
                    return result
                
                # Otherwise, continue BFS
                queue.append((neighbor, distance + 1))
        
        # No matched instance found
        result = (None, float('inf'))
        cache[start_instance] = result
        return result
    
    def analyze_missing_instances(self):
        """Run BFS analysis for all missing instances in deterministic order."""
        print("Analyzing missing instances...")
        
        # Sort missing instances for deterministic results
        sorted_missing_instances = sorted(self.missing_instances)
        
        for i, missing_instance in enumerate(sorted_missing_instances):
            if i % 10000 == 0:
                print(f"Processing {i}/{len(sorted_missing_instances)} missing instances...")
            
            # Find nearest matched instance in forward direction
            forward_match, forward_dist = self.bfs_find_nearest_matched(
                missing_instance, self.forward_graph, self.forward_cache
            )
            
            # Find nearest matched instance in reverse direction
            reverse_match, reverse_dist = self.bfs_find_nearest_matched(
                missing_instance, self.reverse_graph, self.reverse_cache
            )
            
            # Get cell type of missing instance
            cell_type = self.dir1_nodes[missing_instance]['cell']
            
            # Determine the nearest match between forward and reverse
            nearest_match = None
            nearest_dist = float('inf')
            nearest_direction = None
            
            # Compare forward and reverse distances to find the nearest
            if forward_match and forward_dist < nearest_dist:
                nearest_match = forward_match
                nearest_dist = forward_dist
                nearest_direction = 'forward'
            
            if reverse_match and reverse_dist < nearest_dist:
                nearest_match = reverse_match
                nearest_dist = reverse_dist
                nearest_direction = 'reverse'
            
            # Store the nearest mapping
            if nearest_match:
                nearest_slack = self.dir2_nodes[nearest_match]['slack']
                self.nearest_mappings.append({
                    'missing_instance': missing_instance,
                    'nearest_matched_instance': nearest_match,
                    'direction': nearest_direction,
                    'distance': nearest_dist,
                    'slack': nearest_slack,
                    'slack_negative_flag': nearest_slack < 0,
                    'cell_type': cell_type,
                    'forward_match': forward_match if forward_match else 'None',
                    'forward_distance': forward_dist if forward_dist != float('inf') else 'inf',
                    'reverse_match': reverse_match if reverse_match else 'None', 
                    'reverse_distance': reverse_dist if reverse_dist != float('inf') else 'inf'
                })
            
            # Store results for forward direction (keeping original logic)
            if forward_match:
                slack = self.dir2_nodes[forward_match]['slack']
                self.results.append({
                    'instance': missing_instance,
                    'matched_instance': forward_match,
                    'direction': 'forward',
                    'distance': forward_dist,
                    'slack': slack,
                    'slack_negative_flag': slack < 0,
                    'cell_type': cell_type
                })
            
            # Store results for reverse direction
            if reverse_match:
                slack = self.dir2_nodes[reverse_match]['slack']
                self.results.append({
                    'instance': missing_instance,
                    'matched_instance': reverse_match,
                    'direction': 'reverse',
                    'distance': reverse_dist,
                    'slack': slack,
                    'slack_negative_flag': slack < 0,
                    'cell_type': cell_type
                })
    
    def compute_statistics(self):
        """Compute and display statistics."""
        if not self.results:
            print("No results to analyze!")
            return
        
        df_results = pd.DataFrame(self.results)
        
        # Filter out infinite distances
        finite_results = df_results[df_results['distance'] != float('inf')]
        
        if finite_results.empty:
            print("No finite distances found!")
            return
        
        # Overall statistics
        distances = finite_results['distance']
        min_dist = distances.min()
        max_dist = distances.max()
        avg_dist = distances.mean()
        median_dist = distances.median()
        
        # Percentage with negative slack
        negative_slack_count = finite_results['slack_negative_flag'].sum()
        total_count = len(finite_results)
        pct_negative_slack = (negative_slack_count / total_count) * 100
        
        print("\n" + "="*60)
        print("MAPPING ANALYSIS RESULTS")
        print("="*60)
        print(f"Total results with finite distances: {len(finite_results)}")
        print(f"Min distance: {min_dist}")
        print(f"Max distance: {max_dist}")
        print(f"Average distance: {avg_dist:.2f}")
        print(f"Median distance: {median_dist}")
        print(f"Percentage with negative slack: {pct_negative_slack:.2f}%")
        
        # Cell type analysis
        print("\n" + "-"*40)
        print("CELL TYPE ANALYSIS (sorted by avg hop distance)")
        print("-"*40)
        
        cell_type_stats = finite_results.groupby('cell_type')['distance'].agg(['mean', 'count']).reset_index()
        cell_type_stats = cell_type_stats.sort_values('mean')
        
        # for _, row in cell_type_stats.iterrows():
        #     print(f"{row['cell_type']}: avg_distance={row['mean']:.2f}, count={row['count']}")
        
        # Additional analysis excluding BUF and INV cells
        print("\n" + "-"*60)
        print("TOP 10 CELL TYPES BY AVERAGE DISTANCE (Highest First, excluding BUF/INV)")
        print("-"*60)
        
        # Filter out BUF and INV cells
        non_buf_inv_stats = cell_type_stats[
            ~cell_type_stats['cell_type'].str.startswith(('BUF', 'INV'))
        ].copy()
        
        top_by_distance = non_buf_inv_stats.sort_values('mean', ascending=False).head(10)
        for _, row in top_by_distance.iterrows():
            print(f"{row['cell_type']}: avg_distance={row['mean']:.2f}, count={row['count']}")
        
        print("\n" + "-"*60)
        print("TOP 10 CELL TYPES BY COUNT (Highest First, excluding BUF/INV)")
        print("-"*60)
        
        top_by_count = non_buf_inv_stats.sort_values('count', ascending=False).head(10)
        for _, row in top_by_count.iterrows():
            print(f"{row['cell_type']}: count={row['count']}, avg_distance={row['mean']:.2f}")
        
        # Save detailed results
        output_file = "mapping_stats.csv"
        finite_results.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")
        
        # Save summary statistics
        summary_data = {
            'min_distance': [min_dist],
            'max_distance': [max_dist],
            'avg_distance': [avg_dist],
            'median_distance': [median_dist],
            'pct_negative_slack': [pct_negative_slack]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = "distance_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary statistics saved to: {summary_file}")
        
        # Save nearest mappings (one per missing instance)
        if self.nearest_mappings:
            nearest_df = pd.DataFrame(self.nearest_mappings)
            nearest_file = "nearest_mappings.csv"
            nearest_df.to_csv(nearest_file, index=False)
            print(f"Nearest mappings saved to: {nearest_file}")
            print(f"Total missing instances with nearest mappings: {len(self.nearest_mappings)}")
        else:
            print("No nearest mappings found!")
    
    def run_analysis(self):
        """Run the complete analysis."""
        self.load_data()
        self.analyze_missing_instances()
        self.compute_statistics()

def main():
    parser = argparse.ArgumentParser(description="Mapping analysis between two designs")
    parser.add_argument("--dir1", required=True, help="First directory path")
    parser.add_argument("--dir2", required=True, help="Second directory path")
    parser.add_argument("--design", required=True, help="Design name")
    
    args = parser.parse_args()
    
    # Verify directories exist
    if not os.path.exists(args.dir1):
        print(f"Error: Directory {args.dir1} does not exist")
        sys.exit(1)
    
    if not os.path.exists(args.dir2):
        print(f"Error: Directory {args.dir2} does not exist")
        sys.exit(1)
    
    # Run analysis
    analyzer = GraphAnalyzer(args.dir1, args.dir2, args.design)
    analyzer.run_analysis()

if __name__ == "__main__":
    # Default test case as specified in README
    if len(sys.argv) == 1:
        print("Running with default test case...")
        dir1 = "/home/fetzfs_projects/NetlistTomography/sakundu/iccad24_tomography_testcases/ca53_gf12_run2/run2_syn_cp_450_util_0.70/design_post_synth"
        dir2 = "/home/fetzfs_projects/NetlistTomography/sakundu/iccad24_tomography_testcases/ca53_gf12_run2/run2_syn_cp_450_util_0.70/design_place_opt"
        design = "ca53_cpu"
        
        analyzer = GraphAnalyzer(dir1, dir2, design)
        analyzer.run_analysis()
    else:
        main()
