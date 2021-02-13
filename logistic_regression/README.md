

## Logistic regression experiments

### Dependencies
- Python 3.6+
- MPI4PY

### Supported algorithms
- L-SVRG
- VR-NatDIANA
- VR-IntDIANA
- VR-HintDIANA

### Splitting the dataset
```
python preprocessing.py --data a5a --cond 1e-30 --it_max 5000 --n 12
```
### Run the 4 algorithms
```
python main.py --alpha adaptive --beta 0.0 --sigma_Q .001 --data a5a --alg VR-IntDIANA --n 12 
&& python main.py --alpha adaptive --beta 0.0 --sigma_Q .001 --data a5a --alg VR-HintDIANA --n 12 
&& python main.py --alpha adaptive --beta 0.0 --sigma_Q .001 --data a5a --alg L-SVRG --n 12 
&& python main.py --alpha adaptive --beta 0.0 --sigma_Q .001 --data a5a --alg VR-NatDIANA --n 12 
```