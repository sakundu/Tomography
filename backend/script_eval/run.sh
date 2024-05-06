# This script was written and developed by xxx. However, the underlying commands and reports are copyrighted by Cadence. 
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
#!/bin/bash
module unload genus
module load genus/21.1
module unload innovus
module load innovus/21.1

#
# To run the Physical Synthesis (iSpatial) flow - flow2
export PHY_SYNTH=0
export PROJ_DIR=""
export DESIGN="$1"
export TECH="$2"
export DESIGN_DIR="$3"
export ROUTE_BLKG="$4"

mkdir log -p
innovus -64 -overwrite -log log/innovus.log -files run_invs.tcl
