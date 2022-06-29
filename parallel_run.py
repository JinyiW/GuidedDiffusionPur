import os
import argparse
import pandas as pd 
import os 
import yaml
import math 
import subprocess,shlex

def generate_workers(commands):
    workers = []
    for i in range(len(commands)):
        args_list = shlex.split(commands[i])
        # stdout = open(log_files[i], "a")
        # print('executing %d-th command:\n' % i, args_list)
        p = subprocess.Popen(args_list)
        workers.append(p)

    for p in workers:
        p.wait()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--execute', action='store_true', help='whether to execute')
    parser.add_argument('--device', type=int, default=1, help='num of devices')
    parser.add_argument('--rank', type=int, default=0, help='init rank of all num of devices')
    parser.add_argument('--world_size', type=int, default=8, help='num of parallel')
    parser.add_argument('--config', type=str, default='ImageNet_respace_100.yml',  help='Path for saving running related data.')
    args = parser.parse_args()

    os.environ['MKL_THREADING_LAYER'] = 'GNU' 

    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    world_size = args.world_size
    command = []
    num_per_world = math.ceil(config['structure']['run_samples']/config['structure']['bsize']/world_size)

    for i in range(args.device):
        config['structure']['start_epoch'] = (args.rank+i)*num_per_world
        config['structure']['end_epoch'] = (args.rank+i+1)*num_per_world-1
        config['device']['diff_device']= f'cuda:{i}'
        config['device']['clf_device']= f'cuda:{i}'
        config['device']['rank']= args.rank+i
        yml_name = f'{args.config}_rank:{args.rank+i}.yml'
        with open(os.path.join('configs', yml_name), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        command.append(f'python main.py --config {yml_name}')
    # command.append('wait')
    # command.append(f'python utils/compute_data.py --world_size {world_size} --config {args.config}')

    generate_workers(command)
    # with open('run.sh','w') as f:
    #     f.truncate()
    #     for j in range(args.device):
    #         f.write(command[j]+'\n')
    # os.system('sh run.sh')

    eval_command = f'python utils/compute_data.py --world_size {world_size} --config {args.config}'
    os.system(eval_command)