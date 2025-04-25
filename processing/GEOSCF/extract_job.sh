#!/bin/bash
#SBATCH --job-name=geoscf_extract
#SBATCH --account=s2127
#SBATCH --partition=compute
#SBATCH --qos=allnccs
#SBATCH --constraint=mil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:05:00
#SBATCH --error=/discover/nobackup/projects/jcsda/s2127/barre/GEOS_CF_US/geoscf_extract.%j.err
#SBATCH --output=/discover/nobackup/projects/jcsda/s2127/barre/GEOS_CF_US/geoscf_extract.%j.out

module load python/3.10
source /path/to/venv-mil/bin/activate

srun python /discover/nobackup/projects/jcsda/s2127/barre/GEOS_CF_US/extract_geos_cf_yaml.py \
     --config /discover/nobackup/projects/jcsda/s2127/barre/GEOS_CF_US/config.yaml

