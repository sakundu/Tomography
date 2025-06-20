# Netlist Tomography Clustering and Mapping Scripts

## Overview

The scripts in this directory implement a complete pipeline for:
1. **Graph-based clustering** of netlist instances using logical hierarchy, timing slack and geometric proximity
2. **Multi-parameter sweep analysis** for optimization 
3. **Cross-stage mapping** between synthesis and post-route netlists
4. **Statistical analysis** of missing instances and hop distances

## Core Scripts

### 1. `cluster_mapper.py` - Comprehensive Cluster Mapping Pipeline

**Purpose**: Main orchestration script that performs full parameter sweep clustering and maps results between synthesis and post-route netlists.

**Key Features**:
- Generates clusters for multiple parameter combinations (k-hop, alpha weighting)
- Maps post-route clusters back to synthesis netlist 
- Handles missing instances using nearest neighbor mapping
- Outputs both CSV cluster files and DEF placement files

**Usage**:
```bash
python cluster_mapper.py --design_post_route <path> --design_post_synth <path> --design <name>
```

**Algorithm**:
1. Load synthesis and post-route node data
2. Identify matched vs missing instances between stages
3. For each parameter combination (k, alpha):
   - Generate post-route clustering using Leiden algorithm
   - Map clusters to synthesis using direct matching + nearest neighbors
   - Generate output files (CSV + DEF)
4. Produce comprehensive summary report

### 2. `gen_cluster_v1.py` - Phase 2 Clustering with K-hop Pruning

**Purpose**: Advanced clustering with k-hop neighborhood selection and multiple pruning strategies.

**Key Features**:
- **Edge weighting**: `weight = (avg_length/edge_length) + alpha * (1 - slack/clock_period)`
- K-hop neighborhood expansion from critical (negative slack) seed nodes
- Parameter sweep across k-values and alpha weights
- Performance profiling and timing analysis
- Minimum cluster size filtering

**Usage**:
```bash
python gen_cluster_v1.py <design> <data_dir> --min_cluster_size 200 --k_values 0 1 2 --alpha_values 0.0 0.5 1.0 2.0 --ctypes prune_then_cluster
```

### 3. `mapping_analysis.py` - Cross-Stage Instance Mapping

**Purpose**: Analyzes connectivity between different design stages to find optimal mappings for missing instances.

**Key Features**:
- Multi-source BFS for finding nearest matched instances
- Forward and reverse graph traversal
- Hop distance calculation with caching/memoization
- Cell type analysis and statistics
- Deterministic results with reproducible ordering

**Analysis Output**:
- Distance statistics (min, max, average, median hop counts)
- Cell type analysis excluding buffers/inverters
- Negative slack percentage analysis
- Nearest neighbor mappings for each missing instance

### 4. `run_cluster.py` - Simple Leiden Clustering

**Purpose**: Lightweight implementation of Leiden clustering for basic use cases.

**Key Features**:
- Clustering with hierarchy+slack+location weighting
- Configurable alpha parameter for timing vs geometry balance
- Minimum cluster size filtering

## Input Data Requirements

All scripts expect CSV files with the following structure:

**Node files** (`<design>_nodes.csv`):
- `Instance`: Instance name
- `Cell`: Cell type/library name  
- `Slack`: Timing slack value
- `ClockPeriod`: Clock period constraint
- `PT_X`, `PT_Y`: Physical coordinates

**Edge files** (`<design>_edges.csv`):
- `Source`: Source instance name
- `Sink`: Sink instance name  
- `Slack`: Path timing slack
- `ClockPeriod`: Clock period constraint

## Output Files

### Cluster Files
- `<design>_k<k>_alpha<alpha>_synthesis_clusters.csv`: Instance to cluster mapping
- `<design>_k<k>_alpha<alpha>_synthesis.def`: DEF file for physical design tools

### Analysis Reports
- `<design>_comprehensive_summary.csv`: Parameter sweep results
- `mapping_stats.csv`: Detailed hop distance analysis
- `nearest_mappings.csv`: Missing instance to matched instance mappings

## Key Algorithms

### 1. Edge Weight Calculation
```
weight = (average_length / edge_length) + alpha * (1 - slack / clock_period)
```
- **Geometry component**: Favors shorter connections
- **Timing component**: Weighted by alpha, favors critical paths (negative slack)

### 2. K-hop Neighborhood Selection
1. Identify seed nodes (instances with negative slack)
2. Expand k-hops in both forward and reverse directions
3. Apply clustering to selected subgraph

### 3. Missing Instance Mapping
1. Build forward and reverse connectivity graphs
2. Use BFS to find nearest matched instances in both directions  
3. Select minimum hop distance mapping
4. Cache results for performance

## Dependencies

```
pandas >= 1.3.0
numpy >= 1.21.0
igraph-python >= 0.9.0
leidenalg >= 0.8.0
scikit-learn >= 1.0.0
hdbscan >= 0.8.0
networkx >= 2.6.0
matplotlib >= 3.4.0
```

## Usage Examples

### Full Parameter Sweep
```bash
python cluster_mapper.py \
  --design_post_route /path/to/post_route \
  --design_post_synth /path/to/post_synth \
  --design ca53_cpu \
  --output_dir ./results
```

### Single Parameter Clustering  
```bash
python run_cluster.py ca53_cpu /path/to/data --alpha 1.0 --min_cluster_size 200
```

### Cross-stage Mapping Analysis
```bash
python mapping_analysis.py \
  --dir1 /path/to/synthesis \
  --dir2 /path/to/post_route \
  --design ca53_cpu
```

## Performance Notes

- **Large designs**: Use caching and memoization for BFS operations
- **Memory optimization**: Process instances in batches for very large netlists
- **Deterministic results**: All scripts use sorted iteration for reproducible outputs

## File Organization

```
map/
├── cluster_mapper.py      # Main orchestration script
├── gen_cluster.py         # Core clustering algorithms  
├── gen_cluster_v1.py      # Phase 2 with k-hop selection
├── mapping_analysis.py    # Cross-stage mapping analysis
├── run_cluster.py         # Simple Leiden clustering
└── README.md             # This file
```
