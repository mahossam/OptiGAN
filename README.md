## OptiGAN

This repository contains the code for text generation results 
of the paper:  
[OptiGAN: Generative Adversarial Networks for Goal Optimized Sequence Generation](https://arxiv.org/abs/2004.07534), The 2020 International Joint Conference on Neural Networks (IJCNN), 2020.


## Dependencies
This project uses Python 3.6.x, with the following lib dependencies:
* [Tensorflow 1.4](https://www.tensorflow.org/)
* [Numpy 1.14.1](http://www.numpy.org/)
* [Matplotlib 2.2.0](https://matplotlib.org)
* [Scipy 1.0.0](https://www.scipy.org)
* [NLTK 3.2.3](https://www.nltk.org)
* [tqdm 4.19.6](https://pypi.python.org/pypi/tqdm)


## Instructions
The `experiments` folders contain scripts for starting the different experiments.
For example, to reproduce the `COCO Image Captions` experiments, you can try :
```
cd real/experiments
python coco_lstmgan_pg_baseline_mle_gan.py [job_id] [gpu_id]
```
or `EMNLP2017 WMT News`:
```
cd real/experiments
python3 emnlp_small_lstmgan_pg_baseline_mle_gan.py [job_id] [gpu_id]
```
Note to replace [job_id] and [gpu_id] with appropriate numerical values, (0, 0) for example.

## Reference
To cite this work, please use:
```
@INPROCEEDINGS{9206842,
  author={M. {Hossam} and T. {Le} and V. {Huynh} and M. {Papasimeon} and D. {Phung}},
  booktitle={2020 International Joint Conference on Neural Networks (IJCNN)}, 
  title={{OptiGAN: Generative Adversarial Networks for Goal Optimized Sequence Generation}}, 
  year={2020},
  volume={},
  number={},
  pages={1-8}
}
```

## Acknowledgement
This code is based on [RELGAN](https://github.com/weilinie/RelGAN) and the previous benchmarking platform [Texygen](https://github.com/geek-ai/Texygen). 