#!/bin/bash
export K="$1"
export SIZE="$2"
export CONFIG="$3"
export DETAILS="$4"

run_type=`cut -d'_' -f1 <<< $CONFIG`
run_id=`cut -d'_' -f2 <<< $CONFIG`

sub_id=`bc <<< $run_id/100`
sub_dir="run_sub_${sub_id}"

run_dir="./${run_type}/${sub_dir}"
mkdir -p $run_dir
run_dir="./${run_type}/${sub_dir}/run_${run_id}"
script_dir="./ref_script"
cp -rf $script_dir $run_dir

cd $run_dir
./gen_data.sh $K $SIZE $CONFIG "$DETAILS"

