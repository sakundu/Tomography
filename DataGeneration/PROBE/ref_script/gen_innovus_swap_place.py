import random
import sys
import os
from typing import List

def global_swap(board:List[List[List[int]]], k:int, seed:int = 42) -> List[List[List[int]]]:
    random.seed(seed)
    neighbor = [[0, 1], [1,0], [-1, 0], [0, -1]]
    num_rows = len(board)
    num_columns = len(board[0])
    inst_count = num_rows * num_columns
    swap_count = 0
    while swap_count < k*inst_count:
        rc = random.randrange(num_columns)
        rr = random.randrange(num_rows)
        rd = random.sample(neighbor,1)
        nc = rc + rd[0][1]
        nr = rr + rd[0][0]
        if nc >= 0 and nc < num_columns and nr >= 0 and nr < num_rows:
            tmp = board[rr][rc]
            board[rr][rc] = board[nr][nc]
            board[nr][nc] = tmp
            swap_count += 1
    return board

def check_valid_sub_block(size_x, size_y, sub_x, sub_y, size) -> bool:
    if sub_x >= 0 and sub_x + size <= size_x \
        and sub_y >= 0 and sub_y + size <= size_y \
        and size > 0:
        return True
    else:
        return False

def local_swap(board:List[List[List[int]]], sub_x, sub_y, size, k, seed:int = 42) -> List[List[List[int]]]:
    random.seed(seed)
    print(f"#K:{k} SUB_X:{sub_x} SUB_Y:{sub_y} SUB_SIZE:{size}")
    neighbor = [[0, 1], [1,0], [-1, 0], [0, -1]]
    num_rows = len(board)
    num_columns = len(board[0])
    if not check_valid_sub_block(num_columns, num_rows, sub_x, sub_y, size):
        print("Error: sub block is not correct")
        exit()
    
    swap_count = 0
    while swap_count < (k)*size**2:
        rc = random.randrange(sub_x, sub_x + size)
        rr = random.randrange(sub_y, sub_y + size)
        rd = random.sample(neighbor,1)
        nc = rc + rd[0][1]
        nr = rr + rd[0][0]
        if nc >= sub_x and nc < sub_x + size and nr >= sub_y \
                and nr < sub_y + size:
            tmp = board[rr][rc]
            board[rr][rc] = board[nr][nc]
            board[nr][nc] = tmp
            swap_count += 1
    
    return board

def print_borad(board:List[List[List[int]]], inst_height:float,
                inst_width:float, util:float, x_offset:float = 0.0,
                y_offset:float = 0.0, orientation:List[str] = ['R0', 'MX']):
    no_row = len(board)
    no_col = len(board[0])
    x_pitch = inst_width*1.0/util
    y_pitch = inst_height
    for r in range(no_row):
        for c in range(no_col):
            cr = board[r][c][0] + 1
            cc = board[r][c][1] + 1
            cell_name = 'g_'+str(cr)+'_'+str(cc)
            llx = x_offset + x_pitch*c
            lly = y_offset + y_pitch*r
            orient = orientation[(r+1)%2]
            print(f'placeInstance {cell_name} {llx} {lly} {orient} -placed')

if __name__ == '__main__':
    inst_height = float(sys.argv[1])
    inst_width = float(sys.argv[2])
    util = float(sys.argv[3])
    ## Main design details
    no_row = int(sys.argv[4])
    no_col = int(sys.argv[5])
    k = int(sys.argv[6])
    seed = int(os.getenv('SEED', 42))
    ## Create the board
    board = [[[y,x] for x in range(no_col)] for y in range(no_row)]
    board = global_swap(board, k, seed)
    ## Sub block details
    total_argument = len(sys.argv)
    i = 7
    print(f"#K:{k} H:{inst_height} W:{inst_width} U:{util} Seed:{seed}")
    while (total_argument - i)%4 == 0 and i < total_argument:
        sub_x = int(sys.argv[i])
        sub_y = int(sys.argv[i+1])
        size = int(sys.argv[i+2])
        k_sub = float(sys.argv[i+3])
        board = local_swap(board, sub_x, sub_y, size, k_sub, seed)
        i += 4
    
    print_borad(board, inst_height, inst_width, util)
    