import os
import argparse
import pandas as pd 
import os 
import yaml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def save_result(new_row):
    # df_columns = ["Step_size", "Iteration", "Conditional guide", "Path",   "guide_exp_a",  "guide_exp_b",\
    # "guide_scale",  "guide_scale_base", "accurancy", "count_att", "count_diff"]

    # df = pd.DataFrame(columns=df_columns)
    df = pd.read_csv('./logs/result.csv',index_col=0)
    # df = pd.DataFrame()
    df = df.append(new_row, ignore_index=True)
    df.to_csv('./logs/result.csv')
    print('Evalutaion Completed!')

def save_cifar10c_result(new_row):
    # df_columns = ["Step_size", "Iteration", "Conditional guide", "Path",   "guide_exp_a",  "guide_exp_b",\
    # "guide_scale",  "guide_scale_base", "accurancy", "count_att", "count_diff"]

    # df = pd.DataFrame(columns=df_columns)
    # df = pd.read_csv('./logs/cifar10c_result.csv',index_col=0)
    df = df.append(new_row, ignore_index=True)
    df.to_csv('./logs/cifar10c_result.csv')

def cal_mean(config, df,log):
    # calculate mean row
    # df = pd.read_csv(filename,index_col=0) 
    df_acc = df.mean()
    df_sum = df.sum()
    new_row = \
    {
        "Epoch": 'mean',
        "nData": None,
        "att_time": None,
        "pur_time": None,
        "clf_time": None,
        "std_acc": df_acc["std_acc"],
        "att_acc": df_acc["att_acc"],
        "pur_acc_l": df_acc["pur_acc_l"],
        "pur_acc_s": df_acc["pur_acc_s"],
        "pur_acc_o": None,
        "pur_acc_list_l": None,
        "pur_acc_list_s": None,
        "pur_acc_list_o": None,
        "count_att": df_sum["count_att"],
        "count_diff": df_sum["count_diff"]
    }
    df = df.append(new_row, ignore_index=True)
    # df.to_pickle(os.path.join(self.args.log, "result.pkl"))
    df.to_csv(os.path.join(log, "result_all_devices.csv"))

    if config.structure.dataset == "CIFAR10-C":
        file_list = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', \
            'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
        save_cifar10c_result({
            "CIFARC_CLASS": file_list[config.structure.CIFARC_CLASS-1],
            "CIFARC_SEV": config.structure.CIFARC_SEV,
            "Step_size": config.purification.purify_step, 
            "Iteration": config.purification.max_iter, 
            "Conditional guide": "True" if config.purification.cond else "False",
            "Guide Mode": config.purification.guide_mode if config.purification.cond else 0,
            "Path": config.purification.path_number,   
            "guide_scale": config.purification.guide_scale if config.purification.cond else 0,
            "guide_scale_base": config.purification.guide_scale_base if config.purification.cond else 0,
            "standard": round(df_acc["att_acc"],2),
            "accurancy": round(df_acc["pur_acc_l"],2),
            "count_att": df_sum["count_att"],
            "count_diff": df_sum["count_diff"]
            })
    else:
        save_result({"Step_size": config.purification.purify_step, 
            "Respace": config.net.timestep_respacing,
            "Eps": config.attack.ptb,
            "Iteration": config.purification.max_iter, 
            "Conditional guide": "True" if config.purification.cond else "False",
            "Guide Mode": config.purification.guide_mode if config.purification.cond else 0,
            "Path": config.purification.path_number,   
            "guide_scale": config.purification.guide_scale if config.purification.cond else 0,
            "sample_number": config.structure.run_samples,
            "bsize": config.structure.bsize,
            # "guide_scale_base": config.purification.guide_scale_base if config.purification.cond else 0,
            "accurancy": round(df_acc["pur_acc_l"],2),
            "count_att": df_sum["count_att"],
            "count_diff": df_sum["count_diff"]
            })
    

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def compute_data(config,world_size):
    config = dict2namespace(config)

    if config.purification.cond:
        log = os.path.join("logs","{}_{}_COND:{}".format(
            config.structure.dataset, 
            str(config.attack.attack_method),
            config.purification.guide_mode
            ),
            "step_{}_iter_{}_path_{}_per={}_{}".format(
            config.purification.purify_step,
            config.purification.max_iter, 
            config.purification.path_number,
            config.attack.ptb,
            f'{config.purification.guide_scale}+{config.purification.guide_scale_base}'
            ))  
    else: 
        log = os.path.join("logs","{}_{}".format(
            config.structure.dataset, 
            str(config.attack.attack_method)
            ),
            "step_{}_iter_{}_path_{}_per={}".format(
            config.purification.purify_step,
            config.purification.max_iter, 
            config.purification.path_number,
            config.attack.ptb
            ))
    
    # Save current config to the log directory
    with open(os.path.join(log, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False) # create a yaml file 
    
    # concat dataframe
    df_temp = []
    for i in range(world_size):
        df_temp.append(pd.read_csv(os.path.join(log, f"result_{i}.csv"),index_col=0))
    df = pd.concat(df_temp)
    cal_mean(config,df,log)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--execute', action='store_true', help='whether to execute')
    parser.add_argument('--world_size', type=int, default=8, help='num of parallel')
    parser.add_argument('--config', type=str, default='cifar10c1_2.yml',  help='Path for saving running related data.')
    args = parser.parse_args()

    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    compute_data(config,args.world_size)