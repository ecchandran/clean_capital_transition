#!/bin/bash
#SBATCH --job-name=r1_vol_tau_4.17     # create a short name for your job
#SBATCH --nodes=1               # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=evancc@princeton.edu
module purge # note: 10 nodes works, but 30 does not. Also, 10 nodes / 10 tasks each has been deprioritized for a day
module load anaconda3/2024.10
conda activate dmf_env

python model_3.py --regime 1 --tag r1_vol_tau_4.17
 

