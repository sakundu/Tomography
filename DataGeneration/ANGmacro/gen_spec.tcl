source ./ang_lib_setup.tcl

set design tomo_train

foreach lef $lefs {
    read_lef $lef
}

artnetgen_create_spec -num_insts 20000 \
                      -num_primary_ios 1303 \
                      -comb_ratio 0.85 \
                      -avg_bbox 0.1 \
                      -avg_net_degree 2.5 \
                      -avg_topo_order 10.0 \
                      -cell_list ${design}.list \
                      -out_file ${design}.spec

exit
