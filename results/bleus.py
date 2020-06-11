import pandas as pd
import numpy as np
import os, sys
# scores_group_dir = "coco_pg_mle_gan/bleus"
scores_group_dir = "emnlp_pg_mle_gan/bleus"
# scores_group_dir = "emnlp_1020_epochs_old/bleus"

# max_records = 2000//20+1  # for coco
max_records = 1020//20+1  # for emnlp  #1760

dirs = [os.path.join(scores_group_dir, os.path.basename(d[0])) for d in os.walk(scores_group_dir)]
dirs=dirs[1:]

pm = "\\pm"  # u'\u00B1'
def minimum_valid_rows(dataframes_array):
    min_length = np.min([df.values.shape[0] for df in dataframes_array])
    return [df.values[:min_length, :] for df in dataframes_array]

bleus_final_table = list()

for d in dirs:
    bleus2 = list()
    bleus3 = list()
    bleus4 = list()
    bleus5 = list()
    
    directory_bleus_means = list()
    
    files = [fl for _, _, fl in os.walk(d)][0]
    for f in files:
        if "bleus2" in f and "ignore" not in f:
            bleus2.append(f)
        elif "bleus3" in f and "ignore" not in f:
            bleus3.append(f)
        elif "bleus4" in f and "ignore" not in f:
            bleus4.append(f)
        elif "bleus5" in f and "ignore" not in f:
            bleus5.append(f)
    bleus2 = sorted(bleus2)
    bleus3 = sorted(bleus3)
    bleus4 = sorted(bleus4)
    bleus5 = sorted(bleus5)
    
    scores_files = [bleus2, bleus3, bleus4, bleus5]
    for bleu_number, sfs in enumerate(scores_files):
        dfs = [pd.read_csv(os.path.join(d,f), nrows=max_records) for f in sfs]
        bleu_number = bleu_number + 2  
        
        if len(dfs) == 0:
            continue
        # merged_df = pd.concat([dfs[0]]+[pd.DataFrame(dfs[i].values[:, 1], columns=[f'Bleu{bleu_number}_{i+1}']) for i in range(1, len(dfs))], axis=1)
        merged_df = pd.concat([dfs[0]]+[pd.DataFrame(dfs[i].values[:, 1], columns=[os.path.splitext(sfs[i])[0]]) for i in range(1, len(dfs))], axis=1)
    
        rows_trimmed_arrays = minimum_valid_rows(dfs)
        
        means = np.mean(np.concatenate([arr[:, 1:2] for arr in rows_trimmed_arrays], axis=1), axis=1)
        means = np.reshape(means, [-1, 1])
        # mean_df = pd.DataFrame(np.concatenate([dfs[0].values[:, 0:1], means], axis=1), columns=['epoch', dfs[0].columns[1]])
                
        merged_df.to_csv(rf"{d}/{os.path.basename(d)}_bleus{bleu_number}_merged_ignore.csv",index=False)
        
        directory_bleus_means.append([bleu_number, means])
        
    bleus_table = dict()
    bleus_table["Model"] = [f"{os.path.basename(d)}"]
    for saved_means in directory_bleus_means:
        # pd.DataFrame(np.array([np.mean(saved_means[1])*100.0, np.std(saved_means[1])*100.0]).reshape([1, 2]), columns=['Mean', 'Std_dev']).to_csv(rf"{d}/{os.path.basename(d)}_bleus{saved_means[0]}_res_ignore.csv",index=False)
        bleus_table[f"BLEU-{saved_means[0]}"] = [f"{np.mean(saved_means[1])*100.0:2.2f} {pm} {np.std(saved_means[1])*100.0:2.2f}"]
        # bleus_table.append([f"BLEU-{saved_means[0]}", f"{np.mean(saved_means[1])*100.0} {pm} {np.std(saved_means[1])*100.0}"])
    
    if len(bleus_table.keys()) > 1: 
        # pd.DataFrame(bleus_table, columns=['', f'{os.path.basename(d)}']).to_csv(rf"{d}/{os.path.basename(d)}_bleus_res_ignore.csv",index=False)
        bdf = pd.DataFrame(bleus_table)
        # bdf.to_csv(rf"{d}/{os.path.basename(d)}_bleus_res_ignore.csv",index=False)
        bleus_final_table.append(bdf)
        
    
if len(bleus_final_table) > 0:
    pd.concat([df for df in bleus_final_table]).to_csv(rf"{scores_group_dir}/bleus_res.csv",index=False)