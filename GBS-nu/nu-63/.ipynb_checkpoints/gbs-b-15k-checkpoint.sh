#!/bin/bash
#SBATCH --time=168:00:00
#SBATCH -A quantum
#SBATCH --output=15k-fingerprint-n=6=M=6-nbar=5.log
#SBATCH --partition=standard
#SBATCH --mail-user=asa2rc@virginia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=ALL

module load anaconda

#command line format: numSamples, avgPhotons, displacment, maxPhotons, maxModes, dataset to use
time python ../Bin-GBS-Kernel-b-15k-noVac.py 15000 5 0.0 6 6 10