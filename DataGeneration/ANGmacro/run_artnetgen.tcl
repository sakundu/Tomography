source ./ang_lib_setup.tcl

set design tomo_train

foreach lef $lefs {
    read_lef $lef
}

artnetgen_init  -top_module ${design} \
                -spec_file  ${design}.spec \
                -verbose 6 

artnetgen_print_masters

artnetgen_run
artnetgen_write_verilog -out_file ${design}.v 

exit
