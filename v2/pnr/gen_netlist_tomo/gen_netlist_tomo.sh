#!/bin/bash
export DESIGN=$1
export REF_DIR=$(readlink -f "$2")
base_dir=$(readlink -f "$3")
script_dir="/home/fetzfs_projects/NetlistTomography/sakundu/NetlistTomographyCode/gen_netlist_tomo_v1"
mkdir -p ${base_dir}
site_count_hs="-2 -1 0 1 2"
site_count_vs="-1 0 1"
flips="f s"


placer_job_file="${base_dir}/${DESIGN}_place_job"
if [ -f "$placer_job_file" ]; then
  rm -rf $placer_job_file
fi

for ar in $ars; do
  for util in $utils; do
    echo "${script_dir}/run.sh $DESIGN $util $ar $REF_DIR $base_dir" >> $placer_job_file
  done
done

for flip in $flips; do
  for site_count_h in $site_count_hs; do
    for site_count_v in $site_count_vs; do
      echo "${script_dir}/run.sh $DESIGN $site_count_h $site_count_v $flip $REF_DIR $base_dir" >> $placer_job_file
    done
  done
done

user_name=`whoami`

node_file="${base_dir}/${DESIGN}_node"
echo "6/ ${user_name}@hgr" > ${node_file}
echo "6/ ${user_name}@gpl" >> ${node_file}
echo "6/ ${user_name}@dpl" >> ${node_file}
echo "6/ ${user_name}@opc" >> ${node_file}
echo "6/ ${user_name}@dme" >> ${node_file}
echo "6/ ${user_name}@hgr" >> ${node_file}
# echo "4/ ${user_name}@soi" >> ${node_file}

tcsh /home/sakundu/SCRIPT/GNU_PARALLEL/run_gnu_parallel.csh $placer_job_file $node_file
# /home/tool/anaconda/envs/cluster/bin/python ${script_dir}/gen_overlay_cluster.py ${base_dir} $DESIGN
# /home/tool/anaconda/envs/cluster/bin/python ${script_dir}/gen_overlay_cluster_v2.py ${base_dir} $DESIGN
/home/tool/anaconda/envs/cluster/bin/python ${script_dir}/gen_padding_list.py ${base_dir} $DESIGN
