#!/bin/bash
#SBATCH --job-name=DUQ-CIFAR-results
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=16GB                                            


source $HOME/venvs/duq_env/bin/activate

mkdir $TMPDIR/pt
cp -r /scratch/$USER/OOD_project $TMPDIR
cd $TMPDIR/OOD_project
mkdir $TMPDIR/OOD_project/results

echo "Starting training DUQ..."
python -u  main_DUQ_CIFAR.py


mkdir -p /scratch/$USER/DUQ_CIFAR_Results/job_${SLURM_JOBID}

echo "Moving results to $scratch_dir..."
mv $TMPDIR/OOD_project/DUQ_CIFAR_Results /scratch/$USER/DUQ_CIFAR_Results/job_${SLURM_JOBID}
cp /scratch/$USER/OOD_project/slurm-${SLURM_JOBID}.out /scratch/$USER/DUQ_CIFAR_Results/job_${SLURM_JOBID}
echo "Training completed and results moved successfully."