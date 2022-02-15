# IntSGD: Adaptive Floatless Compression of Stochastic Gradients

The code is built on top of the publicly available code of PowerSGD (https://github.com/epfml/powersgd). 

### Training ResNet18 on Cifar-10 dataset

An example of the slurm script is:

```
#!/bin/bash --login
#SBATCH --time=4:00:00
#SBATCH --nodes=8
#SBATCH --gpus-per-node=2
#SBATCH --tasks-per-node=2
#SBATCH --gpus-per-task=1 
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=96G
#SBATCH --partition=batch
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --constraint=v100

module load anaconda3
module load openmpi
module load pytorch

mpirun --np $SLURM_NTASKS python train_intsgd_resnet18.py
```

### Training a 3-layer LSTM on Wikitext-2 dataset

An example of the slurm script is:

```
#!/bin/bash --login
#SBATCH --time=4:00:00
#SBATCH --nodes=8
#SBATCH --gpus-per-node=2
#SBATCH --tasks-per-node=2
#SBATCH --gpus-per-task=1 
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=96G
#SBATCH --partition=batch
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --constraint=v100

module load anaconda3
module load openmpi
module load pytorch

mpirun --np $SLURM_NTASKS python train_intsgd_rand_lstm.py
```
