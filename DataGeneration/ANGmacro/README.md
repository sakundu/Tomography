# How to create ANG-Macro

Here, ANG-Macro means 'macro-integrated artificial netlist'

1. Setup ANG environment
  - Refer to the ANG GitHub repositor \[[GitHub](https://github.com/daeyeon22/artificial_netlist_generator)\]

2. Select macro type and count. In this example, we select 4 types of macros, total 5 instances.
  - u_mem_0: fakeram45_128x32  
  - u_mem_1: fakeram45_1024x39 
  - u_mem_2: fakeram45_256x32  
  - u_mem_3: fakeram45_160x118 
  - u_mem_4: fakeram45_160x118 

3. LEF setting in 'ang_lib_setup.tcl' file
  - Example fakeram LEFs are uploaded on 'macroLEFs' folder
  - Other tech LEF and std. cell LEF can be downloaded from \[[GitHub](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/tree/master/flow/platforms/nangate45/lef)\]
  - Put in tech LEF and std. cell LEF download path into the 'ng45lefdir' of 'ang_lib_setup.tcl' file

4. 



