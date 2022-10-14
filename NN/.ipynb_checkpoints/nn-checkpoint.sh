#!/bin/bash
#SBATCH --time=168:00:00
#SBATCH -A quantum
#SBATCH --output=GBS-NN-2000-5-0-6-9.log
#SBATCH --partition=standard
#SBATCH --mail-user=asa2rc@virginia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=ALL

module load anaconda
module load singularity/3.7.1 pytorch/1.10.0 

#command line format: numSamples, avgPhotons, displacment, maxPhotons, dataset to use
time singularity exec --nv $CONTAINERDIR/pytorch-1.10.0.sif bash run_script.sh