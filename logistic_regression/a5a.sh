#!/bin/bash
python preprocessing.py --data a5a --cond 1e-30 --it_max 5000 --n 12 && python main.py --alpha adaptive --beta 0.0 --sigma_Q .001 --data a5a --alg VR-IntDIANA --n 12 && python main.py --data a5a --alg VR-HintDIANA --n 12 && python main.py --data a5a --alg L-SVRG --n 12 && python main.py --data a5a --alg VR-NatDIANA --n 12 && python plot_figs.py --algs VR-HintDIANA L-SVRG VR-NatDIANA VR-IntDIANA --alphas adaptive adaptive adaptive adaptive --betas 0.0 0.0 0.0 0.0 --data a5a --n 12

