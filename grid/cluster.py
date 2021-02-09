"""Cluster specific setups."""
import math


def bolt(_, num_runs, job_args):
  """
  rm jobs/*.sh jobs/log/* -f && python grid_run.py --grid G --run_name X
  pattern=""; for i in 1 2; do ./kill.sh $i $pattern; done
  ./start.sh
  """
  if len(job_args) > 0:
    jobs_0 = job_args[0]
  else:
    jobs_0 = ['machine0_gpu0']

  # Number of parallel jobs on each machine
  # validate start.sh
  if len(job_args) > 1:
    njobs = job_args[1]
  else:
    njobs = [3]*7 + [2]*2 + [0, 2]
  jobs = []
  for s, n in zip(jobs_0, njobs):
    jobs += ['%s_job%d' % (s, i) for i in range(n)]
  parallel = False  # each script runs in sequence
  print(num_runs)
  return jobs, parallel


def slurm(sargs, num_runs, _):
  """
  rm jobs/*.sh jobs/log/* -f &&\
  python -m grid_run --cluster slurm--grid G --run_name X \
  --task_per_job 1 --job_limit 12 --partition p100,t4
  sbatch jobs/slurm.sbatch
  squeue -u <user>
  scancel -u <user>
  """
  njobs = int(math.ceil(num_runs/sargs.task_per_job))
  ntasks = sargs.job_limit
  partition = sargs.partition
  qos = sargs.qos
  ncpu = sargs.ncpu
  ngpu = sargs.ngpu
  jobs = [str(i) for i in range(njobs)]
  sbatch_f = """#!/bin/bash

#SBATCH --job-name=array
#SBATCH --output=jobs/log/array_%A_%a.log
#SBATCH --array=0-{njobs}%{ntasks}
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:{ngpu}        # Number of GPUs (per node)
#SBATCH -c {ncpu}            # Number of CPUs
#SBATCH --mem=2G
#SBATCH -p {partition}
#SBATCH --ntasks=1
#SBATCH --qos={qos}

date; hostname; pwd
python -c "import jax; print(jax.__version__)"
(while true; do nvidia-smi; top -b -n 1 | head -20; sleep 10; done) &

# the environment variable SLURM_ARRAY_TASK_ID contains
# the index corresponding to the current job step
source $HOME/export_jax.sh
bash jobs/$SLURM_ARRAY_TASK_ID.sh
""".format(njobs=njobs-1, ntasks=ntasks, partition=partition, qos=qos,
           ncpu=ncpu, ngpu=ngpu)
  with open('jobs/slurm.sbatch', 'w') as f:
    print(sbatch_f, file=f)
  parallel = True  # each script runs in parallel
  print('Total jobs: %d, Array jobs: %d, Max active: %d, Partition: %s'
      % (num_runs, njobs, min(njobs, ntasks), partition))
  return jobs, parallel
