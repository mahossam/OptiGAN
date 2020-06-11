import pandas as pd
import numpy as np
import os, sys
# scores_group_dir = "coco_pg_mle_gan/nlls"
scores_group_dir = "emnlp_pg_mle_gan/nlls"
# scores_group_dir = "emnlp_1020_epochs_old/nlls"

# max_records = 2000//20+15+1  # for coco
max_records = 1020//20+15+1  # for emnlp  #1760

dirs = [os.path.join(scores_group_dir, os.path.basename(d[0])) for d in os.walk(scores_group_dir)]
dirs=dirs[1:]


pm = "\\pm"  # u'\u00B1'
def minimum_valid_rows(dataframes_array):
    min_length = np.min([df.values.shape[0] for df in dataframes_array])
    print(f"minimum_valid_rows found = {min_length}")
    return [df.values[:min_length, :] for df in dataframes_array]

nlls_final_table = list()

for d in dirs:
    nll = list()
    
    directory_nlls_means = list()
    
    files = [fl for _, _, fl in os.walk(d)][0]
    for f in files:
        if "ignore" not in f:
            nll.append(f)
    nll = sorted(nll)
    
    scores_files = [nll]
    for _, sfs in enumerate(scores_files):
        dfs = [pd.read_csv(os.path.join(d,f), nrows=max_records) for f in sfs]  # , header=None
        
        if len(dfs) == 0:
            continue
        merged_df = pd.concat([pd.DataFrame(dfs[0].values[:, 1], columns=[os.path.splitext(sfs[0])[0]])] + [pd.DataFrame(dfs[i].values[:, 1], columns=[os.path.splitext(sfs[i])[0]]) for i in range(1, len(dfs))], axis=1)
    
        rows_trimmed_arrays = minimum_valid_rows(dfs)
        
        means = np.mean(np.concatenate([arr[:, 1:2] for arr in rows_trimmed_arrays], axis=1), axis=1)
        means = np.reshape(means, [-1, 1])
        # mean_df = pd.DataFrame(np.concatenate([dfs[0].values[:, 0:1], means], axis=1), columns=['epoch', dfs[0].columns[1]])
                
        merged_df.to_csv(rf"{d}/{os.path.basename(d)}_nlls_merged_ignore.csv",index=False)
        
        directory_nlls_means.append([means])
        
    nlls_table = dict()
    nlls_table["Model"] = [f"{os.path.basename(d)}"]
    for saved_means in directory_nlls_means:
        # chosen_value = saved_means[0][-1][0]
        chosen_value = np.mean(saved_means[0])
        nlls_table["NLL"] = [f"{chosen_value:2.3f} {pm} {np.std(saved_means[0]):2.3f}"]
    
    if len(nlls_table.keys()) > 1: 
        bdf = pd.DataFrame(nlls_table)
        # bdf.to_csv(rf"{d}/{os.path.basename(d)}_nlls_res_ignore.csv",index=False)
        nlls_final_table.append(bdf)
        
    
if len(nlls_final_table) > 0:
    pd.concat([df for df in nlls_final_table]).to_csv(rf"{scores_group_dir}/nlls_res.csv",index=False)
