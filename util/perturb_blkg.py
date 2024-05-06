import os
import re
import sys
import random
from generate_drc_blockage_config import gen_layer_wise_blockage_helper
# sys.path.append("")

def perturb_blkg_helper(line:str, gcell:float = 1.4):
    if random.random() <= 0.5:
        return line
    
    layer = re.search(r"-layer\s+(\S+)", line)
    ## if has -box {x1 y1 x2 y2}
    if re.search(r"-box\s+{(\S+)\s+(\S+)\s+(\S+)\s+(\S+)}", line) is not None:
        bbox = re.search(r"-box\s+{(\S+)\s+(\S+)\s+(\S+)\s+(\S+)}", line)
    else:
        bbox = re.search(r"-box\s+\[list\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\]", line)
    density = re.search(r"-partial\s+(\S+)", line)

    layer_name = layer.group(1)
    bbox = [float(bbox.group(1)), float(bbox.group(2)), float(bbox.group(3)), float(bbox.group(4))]
    density = int(density.group(1))
    # random_value = [-2*gcell, -1*gcell, 0, gcell, 2*gcell]
    
    ## Move box horizontally
    # dx = random.choice(random_value)
    # bbox = [bbox[0]+dx, bbox[1], bbox[2]+dx, bbox[3]]
    
    ## Move box vertically
    # dy = random.choice(random_value)
    # bbox = [bbox[0], bbox[1]+dy, bbox[2], bbox[3]+dy]
    
    ## Change height of the box
    # dh = random.choice(random_value)
    # bbox = [bbox[0], bbox[1], bbox[2], bbox[3]+dh]
    
    ## Change width of the box
    # dw = random.choice(random_value)
    # bbox = [bbox[0], bbox[1], bbox[2]+dw, bbox[3]]
    
    ## Change density
    new_density = min(random.randint(density-2, density+2), 100)
    new_density = max(new_density, 0)
    
    lcb = '{'
    rcb = '}'
    box_str = " ".join([str(round(i, 6)) for i in bbox])
    new_line = f"createRouteBlk -layer {layer_name} -box {lcb}{box_str}{rcb} -partial {new_density}\n"
    return new_line
    
def perturb_blkg(blkg_path, gcell:float = 1.4, count:int = 5, seed:int = 0):
    # random.seed(seed)
    ## Ensure blkg_path is a valid file
    if not os.path.isfile(blkg_path):
        print(f"ERROR: {blkg_path} is not a valid file.")
        return
    
    blkg_file_name = os.path.basename(blkg_path).split(".")[0]
    blkg_file_dir = os.path.dirname(blkg_path)
    new_fp = []
    for i in range(count):
        new_fp.append(open(f"{blkg_file_dir}/{blkg_file_name}_{i}.tcl", "w"))
    
    blkg_fp = open(blkg_path, "r")
    for line in blkg_fp:
        for i in range(count):
            new_fp[i].write(perturb_blkg_helper(line, gcell))
    
    blkg_fp.close()
    for i in range(count):
        new_fp[i].close()
    print(f"Successfully perturbed {blkg_path} and saved to {blkg_file_dir}")
    return

def perturb_config_helper(line:str, gcell:float = 1.4):
    if random.random() <= 0.5:
        return line
    
    items = line.split()
    bbox = [float(i) for i in items[0:4]]
    layer = items[4]
    config = items[5]
    random_value = [-2*gcell, -1*gcell, 0, gcell, 2*gcell]
    
    # Move box horizontally
    dx = random.choice(random_value)
    bbox = [bbox[0]+dx, bbox[1], bbox[2]+dx, bbox[3]]
    
    # Move box vertically
    dy = random.choice(random_value)
    bbox = [bbox[0], bbox[1]+dy, bbox[2], bbox[3]+dy]
    
    # Change height of the box
    dh = random.choice(random_value)
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]+dh]
    
    # Change width of the box
    dw = random.choice(random_value)
    bbox = [bbox[0], bbox[1], bbox[2]+dw, bbox[3]]
    line = " ".join([str(round(i, 6)) for i in bbox]) + f" {layer} {config}\n"
    return line

def perturb_config(config_path, gcell:float = 1.4, count:int = 5, seed:int = 42):
    random.seed(seed)
    ## Ensure config_path is a valid file
    if not os.path.isfile(config_path):
        print(f"ERROR: {config_path} is not a valid file.")
        return
    
    config_file_name = os.path.basename(config_path).split(".")[0]
    config_file_dir = os.path.dirname(config_path)
    new_fp = []
    for i in range(count):
        new_fp.append(open(f"{config_file_dir}/{config_file_name}_{i}.txt", "w"))
    
    config_fp = open(config_path, "r")
    for line in config_fp:
        for i in range(count):
            new_fp[i].write(perturb_config_helper(line, gcell))
    
    config_fp.close()
    for i in range(count):
        new_fp[i].close()
        gen_layer_wise_blockage_helper(f"{config_file_dir}/{config_file_name}_{i}.txt",
                                       f"{config_file_dir}/route_blkg_{i}.tcl")
        perturb_blkg(f"{config_file_dir}/route_blkg_{i}.tcl", gcell, 1)
    
    ## Generate routing blockages for each perturbed config
    
    # gen_layer_wise_blockage_helper()
    
    return

if __name__ == "__main__":
    # perturb_blkg("", count=5, seed=0)
