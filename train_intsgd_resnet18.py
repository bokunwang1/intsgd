import sys
sys.path.append("..")
import train
import os

train.config["n_workers"] = os.getenv("OMPI_COMM_WORLD_SIZE")
train.config["rank"] = os.getenv("OMPI_COMM_WORLD_RANK")
train.config["local_rank"] = os.getenv("OMPI_COMM_WORLD_LOCAL_RANK")
train.config["local_world_size"] = os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE")
train.config["optimizer_reducer"] = "IntQuantReducer"
train.config["optimizer_conv_learning_rate"] = 0.1
train.config["optimizer_learning_rate"] = 0.1
train.config["optimizer_memory"] = False # without error feedback
train.config["optimizer_reducer_int"] = True
train.config["optimizer_mom_before_reduce"] = True
train.config["optimizer_wd_before_reduce"] = True
train.config["optimizer_reducer_rand_round"] = False
train.config["optimizer_overflow_handling"] = True
train.config["optimizer_scale_lr_with_factor"] = 32
train.config["seed"] = 0

train.config["distributed_init_file"] = "../output/dist_init_intsgd"

train.main()
