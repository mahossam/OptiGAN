import json
from time import time
from tqdm import tqdm
from datetime import datetime

# from utils.metrics.Cfg import Cfg
# from utils.metrics.EmbSim import EmbSim
# from utils.metrics.Nll import Nll
from utils.metrics.Bleu_mine import Bleu
# from utils.oracle.OracleCfg import OracleCfg
# from utils.oracle.OracleLstm import OracleLstm
from utils.text_process import *
from utils.utils import *
# import matplotlib.plot as plt
import numpy as np
import pandas as pd
import sys

BLEU_samples = 999999 
#BLEU_samples = 200
bleu_2 = Bleu("",'data/testdata/test_emnlp.txt', 2, sample_size=BLEU_samples)
bleu_3 = Bleu("",'data/testdata/test_emnlp.txt', 3, sample_size=BLEU_samples)
bleu_4 = Bleu("",'data/testdata/test_emnlp.txt', 4, sample_size=BLEU_samples)
bleu_5 = Bleu("",'data/testdata/test_emnlp.txt', 5, sample_size=BLEU_samples)
bleu2_list = list()
bleu3_list = list()
bleu4_list = list()
bleu5_list = list()

# outputs_folder = r"real/experiments/out/coco/20190829/image_coco/multi_False_rl_only_False_meth_2/084751313878/samples"
# outputs_folder = r"real/experiments/out/coco/20190827/image_coco/multi_False_rl_only_False_meth_2/051353173629/samples"
# outputs_folder = r"real/experiments/out/coco/20190827/image_coco/multi_False_rl_only_False_meth_2/051416361014/samples"

# outputs_folder = r"real/experiments/out/coco/20190828/image_coco/gan_only/092017842621/samples"
# outputs_folder = r"real/experiments/out/coco/20190828/image_coco/gan_only/092106003267/samples"
# outputs_folder = r"real/experiments/out/coco/20190828/image_coco/gan_only/222625629470/samples"

# outputs_folder = r"real/experiments/out/20190930/emnlp_news_small/multi_False_rl_only_False_pg_bline_True/101029569707/samples"
outputs_folder = sys.argv[1]
postfix = sys.argv[2]
# outputs_folder = r"real/experiments/out/20190930/emnlp_news_small/multi_False_rl_only_False_pg_bline_True/091310098499/samples"

def evaluate(test_file, epoch):
    bleu_2.test_data = test_file
    bleu_3.test_data = test_file
    bleu_4.test_data = test_file
    bleu_5.test_data = test_file

    bleu2_list.append([epoch, bleu_2.get_bleu()])
    bleu3_list.append([epoch, bleu_3.get_bleu()])
    bleu4_list.append([epoch, bleu_4.get_bleu()])
    bleu5_list.append([epoch, bleu_5.get_bleu()])    


def plot():
    pd.DataFrame(np.array(bleu2_list), columns=['epoch','Bleu2']).to_csv(os.path.join(outputs_folder, f"bleus2_{BLEU_samples}_{postfix}.csv"), index=False)
    pd.DataFrame(np.array(bleu3_list), columns=['epoch','Bleu3']).to_csv(os.path.join(outputs_folder, f"bleus3_{BLEU_samples}_{postfix}.csv"), index=False)
    pd.DataFrame(np.array(bleu4_list), columns=['epoch','Bleu4']).to_csv(os.path.join(outputs_folder, f"bleus4_{BLEU_samples}_{postfix}.csv"), index=False)
    pd.DataFrame(np.array(bleu5_list), columns=['epoch','Bleu5']).to_csv(os.path.join(outputs_folder, f"bleus5_{BLEU_samples}_{postfix}.csv"), index=False)



# print('start pre-train generator:')
# progress_pre = tqdm(range(self.pre_epoch_num))
# for epoch in progress_pre:
# # for epoch in range(self.pre_epoch_num):
#     # start = time()
#     # loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
#     # end = time()
#     # print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
#     # self.add_epoch()
#     if epoch % 5 == 0:
#         # generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
#         # get_real_test_file()
        
#         evaluate()


print('adversarial evalutation:')
adversarial_epoch_num = 2000
progress_adv = tqdm(range(adversarial_epoch_num))        

for epoch in progress_adv:
    # if epoch % 20 == 0 or epoch == adversarial_epoch_num - 1:
    if epoch % 20 == 0:
        evaluate(os.path.join(outputs_folder, "adv_samples_{:05d}.txt".format(epoch)), epoch)


plot()