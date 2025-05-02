#!/bin/bash
base_dir="./"
ref_dir="${base_dir}/ref_dir"
clk_period=$1
util=$2
run_dir="${base_dir}/run_cp_${clk_period}_util_${util}"
cp -r ${ref_dir} ${run_dir}
cd ${run_dir}
./run.sh ${clk_period} ${util}
