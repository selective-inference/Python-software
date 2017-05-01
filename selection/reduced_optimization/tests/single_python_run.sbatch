#!/bin/bash 
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
#set a job name  
#SBATCH --job-name=job
#################  
#a file for job output, you can check job progress, append the job ID with %j to make it unique
#SBATCH --output=jobs/%j.out
#################
# a file for errors from the job
#SBATCH --error=jobs/%j.err
#################
#time you think you need; default is 2 hours
#format could be dd-hh:mm:ss, hh:mm:ss, mm:ss, or mm
#SBATCH --time=30:00:00
#################
#SBATCH --qos=normal
#SBATCH -p normal 
#################
#number of nodes you are requesting, the more you ask for the longer you wait
#SBATCH --nodes=1
#################
#SBATCH --mem=4000

# You can use srun if your job is parallel
#srun R CMD BATCH  ./rtest.R 
# otherwise: 


SEED=$1
DIR=$2

# cd to program directory
cd /home/snigdha/src/selective-inference/selection/reduced_optimization/tests
#cd /Users/snigdhapanigrahi/selective-inference/selection/reduced_optimization/tests

source /home/snigdha/src/selective-inference/.env/bin/activate

python dual_lasso_test.py $SEED $DIR