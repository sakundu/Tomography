import os
import sys
import pandas as pd

def gen_padding_list(base_dir:str, design:str) -> None:
    site_count_hs=[-2, -1, 0, 1, 2]
    site_count_vs=[-1, 0, 1]
    flips=['f', 's']
    node_df = pd.read_csv(f"{base_dir}/{design}_0_0_s/blob_input/{design}_nodes.csv")
    cong_dfs = []
    for site_count_h in site_count_hs:
        for site_count_v in site_count_vs:
            for flip in flips:
                cong_file = f"{base_dir}/{design}_{site_count_h}_{site_count_v}_{flip}/{design}_cong_cells.txt"
                cong_df = pd.read_csv(cong_file)
                cong_dfs.append(cong_df)

    df = pd.concat(cong_dfs, ignore_index=True)
    name_counts = df['Name'].value_counts().reset_index()
    name_counts.columns = ['Name', 'Count']  # Rename for clarity
    # Get unique Name-PinC pairs
    unique_name_pinc = df.drop_duplicates(subset=['Name', 'PinC'])
    # Merge counts back to unique Name-PinC pairs
    result_df = unique_name_pinc.merge(name_counts, on='Name')
    
    # Calculate OnlyLeft, OnlyRight, BothSide
    padding_counts = df.groupby('Name').apply(lambda x: pd.Series({
        'OnlyLeft': ((x['Left'] == 1) & (x['Right'] == 0)).sum(),
        'OnlyRight': ((x['Left'] == 0) & (x['Right'] == 1)).sum(),
        'BothSide': ((x['Left'] == 1) & (x['Right'] == 1)).sum()
    })).reset_index()

    ## Drop Left and Right columns from result_df
    result_df = result_df.drop(columns=['Left', 'Right'])
    result_df = result_df.merge(padding_counts, on='Name')

    ## Merge with node_df based on Name
    result_df = result_df.merge(node_df, on='Name', how='inner')
    # Print Len of Node df and Results df
    print(f"Node df: {len(node_df)} Results df: {len(result_df)}")

    ## Add pin density PinC / Height * Width
    result_df['PinDensity'] = result_df['PinC'] / (result_df['Height'] * result_df['Width'])

    ## Sort Results df based on Count and PinDensity from Large to Small
    result_df = result_df.sort_values(by=['Count', 'PinDensity'], ascending=False).reset_index(drop=True)

    result_df = result_df[['Name','Count','PinDensity', 'OnlyLeft', 'OnlyRight', 'BothSide']]
    
    ## Write out result_df with out header and index
    result_df.to_csv(f"{base_dir}/{design}_cong_cells.csv", index=False, header=False)

if __name__ == '__main__':
    base_dir=sys.argv[1]
    design=sys.argv[2]
    gen_padding_list(base_dir, design)