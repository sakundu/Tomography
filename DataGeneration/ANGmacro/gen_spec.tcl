source ./ang_lib_setup.tcl

set design tomo_train

foreach lef $lefs {
    read_lef $lef
}

artnetgen_create_spec -num_insts 20000 \
                      -num_primary_ios 1919 \
                      -comb_ratio 0.893735 \
                      -avg_bbox 0.1 \
                      -avg_net_degree 2.72062 \
                      -avg_topo_order 16.0 \
                      -cell_list ${design}.list \
                      -out_file ${design}.spec

exit
