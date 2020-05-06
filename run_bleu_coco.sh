#!/bin/bash
. /home/mahmoudm/anaconda3/etc/profile.d/conda.sh
conda activate tf_new_py3


python bleu_post_training.py real/experiments/out/20191107/image_coco/lstmgan_rl_only_False_pg_bline_True/062154687788/samples 0 &
python bleu_post_training.py real/experiments/out/20191107/image_coco/lstmgan_rl_only_False_pg_bline_True/220447863188/samples 1 &
python bleu_post_training.py real/experiments/out/20191107/image_coco/lstmgan_rl_only_False_pg_bline_True/220520707981/samples 2 &
python bleu_post_training.py real/experiments/out/20191106/image_coco/lstmgan_rl_only_False_pg_bline_True/100929620210/samples 3 &

python bleu_post_training.py real/experiments/out/20191106/image_coco/lstmgan_gan_only_True/031829747189/samples 4 &
python bleu_post_training.py real/experiments/out/20191106/image_coco/lstmgan_gan_only_True/051826660762/samples 3 &
python bleu_post_training.py real/experiments/out/20191106/image_coco/lstmgan_gan_only_True/072409891847/samples 2 &
python bleu_post_training.py real/experiments/out/20191106/image_coco/lstmgan_gan_only_True/110629155118/samples 1 &
python bleu_post_training.py real/experiments/out/20191106/image_coco/lstmgan_gan_only_True/110746655869/samples 0 &
