# PROBE data generation

1. Insert PROBE data parameters into ***generate_data_configuration.py***
  - **size**: top region size
  - **sub_sizes_train**: subregion sizes for **training** data
  - **sub_sizes_test**: subregion sizes for **testing** data
  - **counts**: number of subregions
  - **loc_comb**: number of subregion location combinations
  - **k1**: cell swap counts for the whole region
  - **k2**: cell swap counts for each subregion

2. Run *generate_data_configuration.py* for job_file generation
  - Command: ***python generate_data_configuration.py***
  - Output files
    - train_job_file
    - test_job_file

3. Source the job_file in the LINUX shell or submit it using GNU parallel

