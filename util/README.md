## Utility Scripts
In this directory we add various utility scripts for Step-1 and Step-2, data preparation.
- [create_route_blkg_drc_directed.py](./create_route_blkg_drc_directed.py) generates flat and concentric routing blockages for a given hotspot regions. You can also set the centroid of the hotspot region to be the center of the concentric blockages.
- [generate_data_configuration.py](./generate_data_configuration.py) is used to generate the configuration of PROBE-based data for Step-1 and Step-2 model training.
- [generate_drc_blockage_box.py](./generate_drc_blockage_box.py) extracts non-overlapping rectangular routing hotspots from a design.
- [generate_drc_blockage_config.py](./generate_drc_blockage_config.py) generates set of routing blockages for a given non over-lapping rectangular hotspots.
