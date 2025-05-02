# Netlist Tomography
This repository contains scripts for the Netliste Tomography project.

## P&R scripts
Innovus scripts for NOVA design with NG45 technology is uploaded.
```
cd NetlistTomography/pnr
## run target clock period 1.3ns and target utilization 85%
./run_cp_util.sh 1.3 0.85
```

## Cluster scripts
Write out 
(1) Path, edge, and node information
(2) Timing graphs
```
# Open design in Innovus
source scratch.tcl
get_timing_graph_timing_report

source gen_graph_invs.tcl
write_slack_pt_info
write_edge_slack_length_info
```

Generate clustered DEF files
```
# When design name is NOVA and data directory is './'
python gen_cluster.py NOVA .
```
At the end of the above run, you will get:
- {design}_leiden_cluster_path.def
- {design}_hdbscan_cluster_path.def

