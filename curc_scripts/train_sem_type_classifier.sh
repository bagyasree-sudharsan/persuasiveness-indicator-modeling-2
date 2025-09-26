#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=24:00:00
#SBATCH --account=blanca-curc-gpu
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=curc_output/sem_type_classifier.out
#SBATCH --mail-type="ALL"
#SBATCH --mail-user="basu9216@colorado.edu"

# This simply logs your GPU+CUDA 
# information. Verify after the run if
# the GPU was used, the version of CUDA, 
# VRAM allocated, etc. 
nvidia-smi >> logs/nvidia-smi.out

# Activating the conda envrionment. 
# DO NOT USE conda activate, 
# it does not work when submitting
# a job to CURC. 
source /home/${USER}/.bashrc
source ~/.bashrc  

# Making folders for the experiment, 
# if they are not already created. 
mkdir -p metadata
mkdir -p models
mkdir -p logs
mkdir -p results

# transformers library by default  
# downloads model weights to home dir
# (~/.cache/), which only has 2 GB. 
# use TRANSFORMERS_CACHE to select which 
# folder the models should get downloaded to
export TRANSFORMERS_CACHE=metadata/

# Asking CURC's slurm to ensure that 
# CUDA and CUDNN are loaded during run time.
module load cuda
module load cudnn
conda activate conda_nsp_env 
cd /scratch/alpine/basu9216/persuasiveness-indicator-modeling-2
# Run your python file. python -m src.<module>.<script>
# DO NOT ADD `.py` at the end of your script. 
python -m final.sem_type_classifier
