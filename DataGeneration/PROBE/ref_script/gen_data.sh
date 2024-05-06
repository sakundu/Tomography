#!/bin/bash
module unload innovus/21.1
module load innovus/21.1

export K="$1"
export SIZE="$2"
export CONFIG="$3"
export DETAILS="$4"
netlist_dir="../netlists"
input_dir="${netlist_dir}/AOI211_X1_${SIZE}_${SIZE}_knight"
export NETLIST_FILE="${input_dir}/mesh.v"
export DESIGN_SDC="${netlist_dir}/mesh.sdc"
export DESIGN_NAME="mesh"
export UTILIZATION="0.5"

mkdir -p rpt enc log

innovus -64 -init innovus_kth_study.tcl -overwrite -log ./log/kth_study_${K}_${SIZE}_${CONFIG}.log
