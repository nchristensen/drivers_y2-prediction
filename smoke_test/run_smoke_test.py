#! /usr/bin/env python3
import argparse
import os
import socket

import parsl
from parsl.executors.threads import ThreadPoolExecutor
from parsl.executors import HighThroughputExecutor
from parsl.providers import LSFProvider
from parsl.launchers import JsrunLauncher
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_hostname
from parsl.channels import LocalChannel

from parsl.data_provider.files import File
from parsl.app.app import bash_app
from parsl.config import Config

LASSEN_CONDA_ENV = '/g/g20/friedel2/drivers_y2-prediction/emirge/miniforge3/envs/mirgeDriver.Y2prediction'
LASSEN_workdir = '/g/g20/friedel2/work'
ICC_CONDA_ENV = '/home/friedel/scratch/drivers_y2-prediction/emirge/miniforge3/envs/mirgeDriver.Y2prediction'
ICC_workdir = '/home/friedel/scratch/driversy2'


@bash_app
def run_smoketest(fdict=None, mflags="", conda_env=None, stdout='run.stdout', stderr='run.stderr', inputs=[]):
    exec_str = ""
    if conda_env is not None:
        # exec_str += f"conda activate {conda_env}\n"
        exec_str += "source /g/g20/friedel2/drivers_y2-prediction/emirge/miniforge3/bin/activate mirgeDriver.Y2prediction\n"
    exec_str += f"python -u -O -m mpi4py prediction.py"
    # exec_str += f"mpirun -n 1 python -u -O -m mpi4py prediction.py"
    if isinstance(fdict, dict):
        for k, v in fdict.items():
            exec_str += f" {k} {inputs[v].filepath}"
    exec_str += mflags
    exec_str += '\n'
    return exec_str


def parse_args():
    parser = argparse.ArgumentParser(
            description="MIRGE-Com Isentropic Nozzle Driver")
    parser.add_argument("-r", "--restart_file", type=ascii, dest="restart_file",
                        nargs="?", action="store", help="simulation restart file")
    parser.add_argument("-t", "--target_file", type=ascii, dest="target_file",
                        nargs="?", action="store", help="simulation target file")
    parser.add_argument("-i", "--input_file", type=ascii, dest="input_file",
                        nargs="?", action="store", help="simulation config file")
    parser.add_argument("-c", "--casename", type=ascii, dest="casename", nargs="?",
                        action="store", help="simulation case name")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=False,
                        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
                        help="enable lazy evaluation [OFF]")
    parser.add_argument("--overintegration", action="store_true",
                        help="use overintegration in the RHS computations")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    file_dict = {}
    flags = ""
    files = []
    if args.restart_file:
        file_dict["-r"] = len(files)
        if args.restart_file.startswith('/'):
            files.append(File(args.restart_file))
        else:
            files.append(File(os.path.join(os.getcwd(), args.restart_file)))

    if args.target_file:
        file_dict["-t"] = len(files)
        if args.target_file.startswith('/'):
            files.append(File(args.target_file))
        else:
            files.append(File(os.path.join(os.getcwd(), args.target_file)))

    if args.input_file:
        file_dict["-i"] = len(files)
        if args.input_file.startswith('/'):
            files.append(File(args.input_file))
        else:
            files.append(File(os.path.join(os.getcwd(), args.input_file)))
    print(file_dict)
    if args.casename:
        flags += f" -c {args.casename}"
    if args.profile:
        flags += " --profile"
    if args.log:
        flags += " --log"
    if args.lazy:
        flags += " --lazy"
    if args.overintegration:
        flags += " --overintegration"
    host = socket.gethostname()
    if 'lassen' in host:
        lassen_htex = HighThroughputExecutor(label="lassen_htex",
                                             working_dir=LASSEN_workdir,
                                             address='lassen.llnl.gov',  # assumes Parsl is running on a login node
                                             worker_port_range=(50000, 55000),
                                             worker_debug=True,
                                             provider=LSFProvider(
                                                     launcher=JsrunLauncher(
                                                             overrides=f'-g 1 -a 1 -o {LASSEN_workdir}/j$$.stdo -k {LASSEN_workdir}/j$$.stde'),
                                                     walltime="01:00:00",
                                                     nodes_per_block=1,
                                                     init_blocks=1,
                                                     max_blocks=1,
                                                     bsub_redirection=True,
                                                     scheduler_options='#BSUB -q pdebug',
                                                     worker_init=(
                                                             'module load gcc/7.3.1\n'
                                                             'module load spectrum-mpi\n'
                                                             'export XDG_CACHE_HOME="/tmp/$USER/xdg-scratch"\n'
                                                     ),
                                                     project='uiuc',
                                                     cmd_timeout=600
                                             ),
                                             )
        myconfig = Config(executors=[lassen_htex],
                          strategy=None
                          )
        workdir = LASSEN_workdir
    elif 'campuscluster' in host:
        icc_htex = HighThroughputExecutor(label='ICC_HTEX',
                                          working_dir=ICC_workdir,
                                          address=address_by_hostname(),
                                          max_workers=24,
                                          provider=SlurmProvider(partition='secondary',
                                                                 channel=LocalChannel(),
                                                                 nodes_per_block=1,
                                                                 init_blocks=1,
                                                                 worker_init=f'conda activate {ICC_CONDA_ENV}',
                                                                 walltime='01:00:00',
                                                                 launcher=SrunLauncher(overrides='--ntasks-per-node=24')
                                                                 )
                                          )

        myconfig = Config(executors=[icc_htex],
                          internal_tasks_max_threads=2,
                          strategy=None
                          )
        workdir = ICC_workdir
    else:
        local_htex = ThreadPoolExecutor(max_threads=3, label='local_tpex')
        myconfig = Config(executors=[local_htex],
                          internal_tasks_max_threads=2,
                          strategy=None
                          )
        workdir = '/home/friedel/CEESD/drivers_y2-prediction/smoke_test/work'
    parsl.set_stream_logger()

    parsl.clear()
    parsl.load(myconfig)

    test_run = run_smoketest(fdict=file_dict, mflags=flags, stdout=f'{workdir}/smoketest_run.stdout', stderr=f'{workdir}/smoketest_stderr', inputs=files)
    test_run.result()
    print("Done")
