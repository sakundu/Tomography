#!/bin/bash

python connectMems.py fakeram45_128x32  u_mem_0 ./lef/fakeram45_128x32.lef
python connectMems.py fakeram45_1024x39 u_mem_1 ./lef/fakeram45_1024x39.lef
python connectMems.py fakeram45_256x32  u_mem_2 ./lef/fakeram45_256x32.lef
python connectMems.py fakeram45_160x118 u_mem_3 ./lef/fakeram45_160x118.lef
python connectMems.py fakeram45_160x118 u_mem_4 ./lef/fakeram45_160x118.lef
python replaceICGwithSDFF.py
