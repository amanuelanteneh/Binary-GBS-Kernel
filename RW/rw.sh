#!/bin/bash
#SBATCH --time=168:00:00
#SBATCH -A quantum
#SBATCH --output=8-0.001.log
#SBATCH --partition=standard
#SBATCH --mail-user=asa2rc@virginia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=ALL

module load anaconda

#command line format: numSamples, avgPhotons, displacment, maxPhotons, maxModes, dataset to use
time python RW-Kernel.py 3