# IntSGD: Floatless Compression of Stochastic Gradients

This repository includes the inplementations of IntSGD. We compare IntSGD with several representative baselines on both convex and nonconvex problems.

 ## Convex Problem
 
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
