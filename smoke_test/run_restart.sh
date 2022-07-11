#!/bin/bash
mpirun -n 2 python -u -O -m mpi4py prediction.py -i run_params.yaml -r restart_data/prediction-000010 --log
