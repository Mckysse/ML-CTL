#!/bin/bash
source /home/intern/bdchen/.bashrc
conda activate pytorch

cd $REPO
python multi_cl_train_mbert.py


