# Yuncong Ma, 2/12/2024
# pNet
# Provide functions to submit jobs to cluster environment


#########################################
# Packages
from datetime import datetime
import os
from sys import platform

from Module.Data_Input import write_json_setting, load_json_setting, setup_result_folder

#########################################


def setup_cluster(dir_pnet_result: str,
                  dir_env: str,
                  dir_pnet: str,
                  dir_python: str,
                  submit_command='qsub -terse -j y',
                  thread_command='-pe threaded ',
                  memory_command='-l h_vmem=',
                  log_command='-o ',
                  computation_resource=None or dict):

    """
    Setup cluster environment and commands to submit jobs

    :param dir_pnet_result: directory of pNet result folder
    :param dir_env: directory of the desired virtual environment
    :param dir_pnet: directory of the pNet toolbox
    :param dir_python: absolute directory to the python folder, ex. /Users/YuncongMa/.conda/envs/pnet/bin/python
    :param submit_command: command to submit a cluster job
    :param thread_command: command to setup number of threads for each job
    :param memory_command: command to setup memory allowance for each job
    :param log_command: command to specify the logfile
    :param computation_resource: None or a dict which specifies the number of threads and memory for different processes
    :return: None

    Yuncong Ma, 2/12/2024
    """

    dir_pnet_dataInput, _, _, _, _, _ = setup_result_folder(dir_pnet_result)
    setting = {'dir_env': dir_env, 'dir_pnet': dir_pnet, 'dir_python': dir_python, 'submit_command': submit_command, 'thread_command': thread_command, 'memory_command': memory_command, 'log_command': log_command,
               'computation_resource': computation_resource}

    write_json_setting(setting, os.path.join(dir_pnet_dataInput, 'Cluster_Setting.json'))


def submit_bash_job(dir_pnet_result: str,
                    python_command: str or None,
                    memory=50, n_thread=4,
                    logFile=None,
                    bashFile=None,
                    pythonFile=None,
                    create_python_file=True):
    """
    submit a bash job to the desired cluster environment
    Generate bash and python files automatically

    :param dir_pnet_result: directory of pNet result folder
    :param python_command: the Python function to run, with dir_pnet_result as a preset variable
    :param memory: a real number in GB
    :param n_thread: number of threads to use
    :param logFile: full directory of a log file
    :param bashFile: full directory of the bash file to generate
    :param pythonFile: full directory of the python file to generate
    :param create_python_file: bool, create a new Python file or not
    :return: None

    Yuncong Ma, 2/12/2024
    """

    # load cluster setting
    dir_pnet_dataInput, _, _, _, _, _ = setup_result_folder(dir_pnet_result)
    setting = load_json_setting(os.path.join(dir_pnet_dataInput, 'Cluster_Setting.json'))
    dir_env = setting['dir_env']
    dir_pnet = setting['dir_pnet']
    dir_python = setting['dir_python']
    submit_command = setting['submit_command']
    thread_command = setting['thread_command']
    memory_command = setting['memory_command']
    log_command = setting['log_command']

    # current date and time
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

    # create a new bash file
    if os.path.isfile(bashFile):
        os.remove(bashFile)

    bashFile = open(bashFile, 'w')

    # header
    print('#!/bin/sh\n', file=bashFile, flush=True)
    print('# This bash script is to run a pNet job in the desired cluster environment', file=bashFile, flush=True)
    print(f'# created on {date_time}\n', file=bashFile, flush=True)
    print(f'# Use command to submit this job:\n# $ {submit_command} {thread_command}{n_thread} {memory_command}{memory} {log_command}{logFile} {bashFile.name}\n', file=bashFile, flush=True)
    print(f'source activate {dir_env}\n', file=bashFile, flush=True)
    print(r'echo -e "Start time : `date +%F-%H:%M:%S`\n" ', file=bashFile, flush=True)
    print(f'\n{dir_python} {pythonFile}\n', file=bashFile, flush=True)
    print(r'echo -e "Finished time : `date +%F-%H:%M:%S`\n" ', file=bashFile, flush=True)
    bashFile.close()
    bashFile = bashFile.name

    if create_python_file:
        # create a Python job file
        if os.path.isfile(pythonFile):
            os.remove(pythonFile)

        pythonFile = open(pythonFile, 'w')
        print('# This python file is to run a pNet job', file=pythonFile, flush=True)
        print(f'# created on {date_time}\n', file=pythonFile, flush=True)
        print('import sys\nimport os\n', file=pythonFile, flush=True)
        print(f"dir_pnet = '{dir_pnet}'", file=pythonFile, flush=True)
        #print(f"sys.path.append(os.path.join(dir_pnet, 'Python'))", file=pythonFile, flush=True)
        print(f"sys.path.append(dir_pnet)", file=pythonFile, flush=True)  # modified by FY, 07/26/2024
        #print('import pNet\n', file=pythonFile, flush=True)
        print('import pnet\n', file=pythonFile, flush=True)     # mod by hm, 07/19/2024

        print(f"dir_pnet_result = '{dir_pnet_result}'\n", file=pythonFile, flush=True)
        print(f"{python_command}\n", file=pythonFile, flush=True)

        pythonFile.close()

    # execute a shell command to submit a cluster job, only for linux based systems
    if platform == "linux":
        os.system(f'{submit_command} {thread_command}{n_thread} {memory_command}{memory} {log_command}{logFile} {bashFile}')

