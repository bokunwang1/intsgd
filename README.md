# IntSGD: Floatless Compression of Stochastic Gradients

This repository includes the inplementation of IntSGD. We compare IntSGD with several baselines in the SwitchML framework on both convex and nonconvex problems.

 ## 1. Convex Problem
 
We consider the distributed l2-regularized logistic regression problem. As mentioned in paper, the variant of IntSGD: VR-IntDIANA is preferrable here.  
 
To split the data heterogeneously onto 12 local workers and preprocess it, run:
```
cd ./logistic_regression
python preprocessing.py --data a5a --cond 1e-30 --it_max 5000 --n 12
``` 
Then, run intsgd with adaptive rescaling factor on 12 workers:
```
python main.py --alpha adaptive --beta 0.0 --sigma_Q .001 --data a5a --alg VR-IntDIANA --n 12
```

#### Prerequisites:
- Python 3.6+
- MPI4PY

To reproduce the figure in our paper, run:
```
bash a5a.sh
```

## 2. Nonconvex Problems
We consider 2 deep learning tasks here:
- Training convnets (e.g., ResNet18, ResNet50) on cifar10 dataset
- Training VAEs (VAE or beta-TC VAE) on MNIST, Fashion MNIST, and CelebA datasets.

Since we don't have multiple GPUs connected by a switch, the experiment is simulated and run on a single GPU with 8 "fake" workers. Please refer to ``./optimizers/intsgd.py`` for more details.

### Training convnets

Scripts to reproduce our shown results on ResNet18:

```
python main.py --n_epoch 180 --alg SGD --lr0 0.1 --bs 16 --net resnet
python main.py --n_epoch 180 --alg NatSGD --lr0 0.1 --bs 16 --net resnet
python main.py --n_epoch 180 --alg HintSGD --lr0 0.1 --bs 16 --net resnet
python main.py --n_epoch 180  --beta 0.0 --sigma_sq 1e-8  --alg IntSGD --lr0 0.1 --bs 16 --net resnet --q single --r det --alpha0 1.0 
```
Plot the figs that compare IntSGD with SGD:
```
python plot_figs.py --algs SGD IntSGD --sigma_sqs 0.0 1e-08 --betas 0.0 0.0 --lw False False --rr False False --alpha0s 0.0 1.0  --lr0 0.1 --bs 16 --net resnet
```

#### Prerequisites:
- Python 3.6+
- Pytorch 1.0+

For more details, please check our paper.