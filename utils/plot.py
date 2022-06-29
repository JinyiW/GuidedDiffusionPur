import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import yaml
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# matplotlib.rcParams['text.usetex'] = True

def var_vs_cons(df1,df2):
    fig, ax = plt.subplots()
    linsp = df1["Step_size"]
    acc1 = df1["accurancy"]
    num_attack1 = df1["count_att"]
    num_diff1 = df1["count_diff"]
    acc2 = df2["accurancy"]
    num_attack2 = df2["count_att"]
    num_diff2 = df2["count_diff"]

    fig, ax_acc = plt.subplots(figsize=(8, 6))

    cl_acc = 'tab:red'
    ax_acc.plot(linsp, acc1, '-', color=cl_acc, label="Accuracy (constant)")
    ax_acc.plot(linsp, acc2, '-', color='y', label="Accuracy")
    ax_acc.set_xlabel('Diffusion Length')
    # ax_acc.set_xlim([0, len(los_lst)])
    ax_acc.set_ylabel('Accuracy')
    # ax_acc.set_ylim([0, 100])
    ax_acc.tick_params(axis='y', labelcolor=cl_acc)
    ax_acc.grid(True, ls='--')
    # ax_omg.set_title('$\\theta$ and $\omega$ over time')

    ax_los = ax_acc.twinx()
    cl_los = 'tab:blue'
    ax_los.plot(linsp,num_attack1, '*-', color=cl_los, label="caused by attack  (constant)")
    ax_los.plot(linsp,num_diff1, '+-', color=cl_los,label="caused by diffusion  (constant)")
    ax_los.plot(linsp,num_attack2, '*-', color='g', label="caused by attack ")
    ax_los.plot(linsp,num_diff2, '+-', color='g',label="caused by diffusion ")
    ax_los.set_ylabel('Number of Samples', color=cl_los)
    # ax_los.set_ylim([0, 10000])
    ax_los.tick_params(axis='y', labelcolor=cl_los)
    ax_los.grid(False)
    
    lines, labels = ax_acc.get_legend_handles_labels()
    lines2, labels2 = ax_los.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc=0)

    plt.savefig(os.path.join('figures',f"var_vs_cons.pdf"), dpi=800, format="pdf")
    plt.close(fig)

def guide_compare_vs_step(scale_list,df_list):
    font1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        }
    fig, ax = plt.subplots()
    color_list = ['y','g','b','gray']
    fig, ax_acc = plt.subplots(figsize=(8, 2))
    for i in range(len(scale_list)):
        linsp = df_list[i]["Step_size"]
        acc = df_list[i]["accurancy"]
        ax_acc.plot(linsp, acc, '-', color='r', label=f"Accuracy With Guidance")
    cl_acc = 'black'
    linsp = df_list[-1]["Step_size"]
    acc1 = df_list[-1]["accurancy"]
    ax_acc.plot(linsp, acc1, '-', color='b', label="Accuracy Without Guidence")
    plt.tick_params(labelsize=10)

    ax_acc.set_xlabel('Diffusion Length',font1)
    ax_acc.set_xlim([0,300])
    # ax_acc.set_xlim([0, len(los_lst)])
    ax_acc.set_ylabel('Accuracy', font1)
    ax_acc.set_ylim([60, 90])
    ax_acc.tick_params(axis='y', labelcolor=cl_acc)
    # ax_acc.grid(True, ls='--')
    # ax_omg.set_title('$\\theta$ and $\omega$ over time')

    # ax_los = ax_acc.twinx()
    # cl_los = 'tab:blue'
    # ax_los.plot(linsp,num_attack1, '*-', color=cl_los, label="caused by attack")
    # ax_los.plot(linsp,num_diff1, '+-', color=cl_los,label="caused by diffusion")
    # ax_los.plot(linsp,num_attack2, '*-', color='g', label="caused by attack (Guided")
    # ax_los.plot(linsp,num_diff2, '+-', color='g',label="caused by diffusion (Guided")
    # ax_los.set_ylabel('Number of Samples', color=cl_los)
    # # ax_los.set_ylim([0, 10000])
    # ax_los.tick_params(axis='y', labelcolor=cl_los)
    # ax_los.grid(False)
    
    # lines, labels = ax_acc.get_legend_handles_labels()
    # lines2, labels2 = ax_los.get_legend_handles_labels()
    # plt.legend(lines + lines2, labels + labels2, loc=0)

    plt.legend(prop=font1)
    plt.savefig(os.path.join('figures',f"guide_compare.png"), dpi=800, format="png")
    plt.close(fig)


def guide_compare_vs_iter(scale_list,df_list):
    fig, ax = plt.subplots()
    color_list = ['y','g','b','gray']
    fig, ax_acc = plt.subplots(figsize=(8, 6))
    for i in range(len(scale_list)):
        linsp = df_list[i]["Iteration"]
        acc = df_list[i]["accurancy"]
        ax_acc.plot(linsp, acc, '-', color=color_list[i], label=f"Accuracy (Guide scale={scale_list[i]})")
    cl_acc = 'tab:red'
    linsp = df_list[-1]["Iteration"]
    acc1 = df_list[-1]["accurancy"]
    ax_acc.plot(linsp, acc1, '-', color=cl_acc, label="Accuracy (No guidence)")
    
    ax_acc.set_xlabel('Iteration')
    # ax_acc.set_xlim([0, len(los_lst)])
    ax_acc.set_ylabel('Accuracy', color=cl_acc)
    ax_acc.set_ylim([40, 90])
    ax_acc.tick_params(axis='y', labelcolor=cl_acc)

    plt.legend()
    plt.savefig(os.path.join('figures',f"guide_compare.pdf"), dpi=800, format="pdf")
    plt.close(fig)


def acc_vs_iter(df, step):
    fig, ax = plt.subplots()
    linsp = df["Iteration"]
    acc = df["accurancy"]
    num_attack = df["count_att"]/100
    num_diff = df["count_diff"]/100

    fig, ax_acc = plt.subplots(figsize=(8, 6))
    
    cl_acc = 'tab:red'
    ax_acc.plot(linsp, acc, '-', color=cl_acc, label="accuracy")
    ax_acc.plot(linsp,num_attack, '*-', color='b', label="caused by attack")
    ax_acc.plot(linsp,num_diff, '+-', color='g',label="caused by diff")
    ax_acc.set_xlabel('Iteration')
    # ax_acc.set_xlim([0, len(los_lst)])
    ax_acc.set_ylabel('Accuracy')
    # ax_acc.set_ylim([0, 100])
    ax_acc.tick_params(axis='y', labelcolor=cl_acc)
    ax_acc.grid(False)
    # ax_omg.set_title('$\\theta$ and $\omega$ over time')

    # ax_los = ax_acc.twinx()
    # cl_los = 'tab:blue'
    # ax_los.plot(linsp,num_attack, '*-', color=cl_los, label="caused by attack")
    # ax_los.plot(linsp,num_diff, '+-', color=cl_los,label="caused by diff")
    # ax_los.set_ylabel('Number of Samples', color=cl_los)
    # # ax_los.set_ylim([0, 10000])
    # ax_los.tick_params(axis='y', labelcolor=cl_los)
    # ax_los.grid(False)
    
    # lines, labels = ax_acc.get_legend_handles_labels()
    # lines2, labels2 = ax_los.get_legend_handles_labels()
    plt.legend()

    plt.savefig(os.path.join('figures',f"acc_vs_iter(step={step}).png"), dpi=800, format="png")
    plt.close(fig)


def acc_vs_budget_iter(df,num):
    fig, ax = plt.subplots()
    # num_attack = df["count_att"]/100
    # num_diff = df["count_diff"]/100

    fig, ax_acc = plt.subplots(figsize=(8, 6))
    color_list = ['r','y','g','b','gray']
    for i in range(num):
        linsp = df[i]["Step_size"]*(i+1)
        acc = df[i]["accurancy"]
        z = np.polyfit(linsp,acc, 13) 
        p = np.poly1d(z) 
        ax_acc.plot(linsp,p(linsp),color=color_list[i])
        ax_acc.scatter(linsp, acc,s=2, color=color_list[i], label=f"Accuracy Iter={i+1}")
    # ax_acc.plot(linsp,num_attack, '*-', color='g', label="caused by attack")
    # ax_acc.plot(linsp,num_diff, '+-', color='b',label="caused by diff")
    ax_acc.set_xlabel('Step_size*Iteration')
    # ax_acc.set_xlim([0, len(los_lst)])
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_ylim([60, 88])
    ax_acc.tick_params(axis='y')
    ax_acc.grid(False)
    # ax_omg.set_title('$\\theta$ and $\omega$ over time')

    # ax_los = ax_acc.twinx()
    # cl_los = 'tab:blue'
    # ax_los.plot(linsp,num_attack, '*-', color=cl_los, label="caused by attack")
    # ax_los.plot(linsp,num_diff, '+-', color=cl_los,label="caused by diff")
    # ax_los.set_ylabel('Number of Samples', color=cl_los)
    # # ax_los.set_ylim([0, 10000])
    # ax_los.tick_params(axis='y', labelcolor=cl_los)
    # ax_los.grid(False)
    
    # lines, labels = ax_acc.get_legend_handles_labels()
    # lines2, labels2 = ax_los.get_legend_handles_labels()
    # plt.legend(lines + lines2, labels + labels2, loc=0)
    plt.legend()

    plt.savefig(os.path.join('figures',f"Acc_vs_Budget_Iter.pdf"), dpi=800, format="pdf")
    plt.close(fig)

def acc_vs_budget_iter_guide(df,num):
    fig, ax = plt.subplots()
    fig, ax_acc = plt.subplots(figsize=(8, 6))
    color_list = ['r','y','g','b','gray']
    for i in range(num):
        linsp = df[i]["Step_size"]*(i+1)
        acc = df[i]["accurancy"]
        ax_acc.plot(linsp, acc, '-', color=color_list[i], label=f"Accuracy Iter={i+1}")
    ax_acc.set_xlabel('Step_size*Iteration')
    # ax_acc.set_xlim([0, len(los_lst)])
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_ylim([80, 92])
    ax_acc.tick_params(axis='y')
    ax_acc.grid(False)
    plt.legend()
    plt.savefig(os.path.join('figures',f"Acc_vs_Budget_Iter_Guide.pdf"), dpi=800, format="pdf")
    plt.close(fig)

def acc_vs_budget_step(df,num):
    fig, ax = plt.subplots()
    # num_attack = df["count_att"]/100
    # num_diff = df["count_diff"]/100

    fig, ax_acc = plt.subplots(figsize=(8, 6))
    color_list = ['r','y','g','b','gray']
    for i in range(num):
        linsp = df[i]["Iteration"]*(i+1)*10
        acc = df[i]["accurancy"]
        ax_acc.plot(linsp, acc, '-', color=color_list[i], label=f"Accuracy Step={i*10+10}")
    # ax_acc.plot(linsp,num_attack, '*-', color='g', label="caused by attack")
    # ax_acc.plot(linsp,num_diff, '+-', color='b',label="caused by diff")
    ax_acc.set_xlabel('Step_size*Iteration')
    # ax_acc.set_xlim([0, len(los_lst)])
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_ylim([60, 88])
    ax_acc.tick_params(axis='y')
    ax_acc.grid(False)
    plt.legend()

    plt.savefig(os.path.join('figures',f"Acc_vs_Budget_Step.pdf"), dpi=800, format="pdf")
    plt.close(fig)

def acc_vs_iters(df,steps):
    fig, ax = plt.subplots()
    
    fig, ax_acc = plt.subplots(figsize=(8, 4))
    # ax_los = ax_acc.twinx()
    color_list = ['r','y','g','b','gray']
    for i in range(len(steps)):
        linsp = df[i]["Iteration"]
        acc = df[i]["accurancy"]
        num_attack = df[i]["count_att"]/100
        num_diff = df[i]["count_diff"]/100
        ax_acc.plot(linsp, acc, '-', color=color_list[i], label=f"Diffusion Length={steps[i]}")
        # ax_los.plot(linsp,num_attack, '--', color='r', label="caused by attack")
        # ax_los.plot(linsp,num_diff, '--', color='b',label="caused by diff")
    # ax_acc.plot(linsp,num_attack, '*-', color='g', label="caused by attack")
    # ax_acc.plot(linsp,num_diff, '+-', color='b',label="caused by diff")
    font1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        }
    ax_acc.set_xlabel('Iteration',font1)
    # ax_acc.set_xlim([0, len(los_lst)])
    ax_acc.set_ylabel('Accuracy',font1)
    ax_acc.set_ylim([60, 88])
    ax_acc.set_xlim([0, 15])
    ax_acc.tick_params(axis='y')
    ax_acc.grid(False)

    # cl_los = 'tab:blue'
    
    # ax_los.set_ylabel('Number of Samples')
    # ax_los.set_ylim([0, 50])
    # ax_los.tick_params(axis='y')
    # ax_los.grid(False)
    
    # lines, labels = ax_acc.get_legend_handles_labels()
    # lines2, labels2 = ax_los.get_legend_handles_labels()
    plt.legend(prop=font1)

    plt.savefig(os.path.join('figures',f"Acc_vs_Iters.png"), dpi=800, format="png")
    plt.close(fig)


def acc_vs_step(df,iter_num):
    fig, ax = plt.subplots()
    linsp = df["Step_size"]
    acc = df["accurancy"]
    num_attack = df["count_att"]/100
    num_diff = df["count_diff"]/100

    fig, ax_acc = plt.subplots(figsize=(8, 6))

    cl_acc = 'tab:red'
    ax_acc.plot(linsp, acc, '-', color=cl_acc, label="Accuracy")
    ax_acc.plot(linsp,num_attack, '*-', color='g', label="caused by attack")
    ax_acc.plot(linsp,num_diff, '+-', color='b',label="caused by diff")
    ax_acc.set_xlabel('Step_size')
    # ax_acc.set_xlim([0, len(los_lst)])
    ax_acc.set_ylabel('Accuracy', color=cl_acc)
    # ax_acc.set_ylim([0, 100])
    ax_acc.tick_params(axis='y', labelcolor=cl_acc)
    ax_acc.grid(False)
    # ax_omg.set_title('$\\theta$ and $\omega$ over time')

    # ax_los = ax_acc.twinx()
    # cl_los = 'tab:blue'
    # ax_los.plot(linsp,num_attack, '*-', color=cl_los, label="caused by attack")
    # ax_los.plot(linsp,num_diff, '+-', color=cl_los,label="caused by diff")
    # ax_los.set_ylabel('Number of Samples', color=cl_los)
    # # ax_los.set_ylim([0, 10000])
    # ax_los.tick_params(axis='y', labelcolor=cl_los)
    # ax_los.grid(False)
    
    # lines, labels = ax_acc.get_legend_handles_labels()
    # lines2, labels2 = ax_los.get_legend_handles_labels()
    # plt.legend(lines + lines2, labels + labels2, loc=0)
    plt.legend()

    plt.savefig(os.path.join('figures',f"Acc_vs_Steps(Iteration={iter_num}).pdf"), dpi=800, format="pdf")
    plt.close(fig)

def acc_vs_n(df,budget):
    fig, ax = plt.subplots()
    fig, ax_acc = plt.subplots(figsize=(8, 6))
    for i in range(len(budget)):
        ite = df[i]["Iteration"]
        step = df[i]["Step_size"]
        acc = df[i]["accurancy"]
        ax_acc.scatter(ite, step,c=acc,cmap=plt.cm.rainbow, s=1, label=f"Budget={budget[i]}")
    ax_acc.set_xlabel('Iteration')
    # ax_acc.set_xlim([0, len(los_lst)])
    ax_acc.set_ylabel('Step')
    # ax_acc.set_ylim([0, 100])
    ax_acc.tick_params(axis='y')
    ax_acc.grid(False)
    plt.legend()

    plt.savefig(os.path.join('figures',"Acc_vs_N.pdf"), dpi=800, format="pdf")
    plt.close(fig)

def acc_vs_budget(df):
    fig, ax_acc = plt.subplots(figsize=(8, 6))
    df_low = df[(df["accurancy"]<80)]
    x = df_low["Iteration"]
    step = df_low["Step_size"]
    acc = df_low["accurancy"]
    ax_acc.scatter(x, step, c='b' ,s=2)
    df_low = df[(df["accurancy"]>=80)]
    x = df_low["Iteration"]
    step = df_low["Step_size"]
    acc = df_low["accurancy"]
    ax_acc.scatter(x, step,c=acc,cmap=plt.cm.jet, s=2, label=f"Budget=200")
    ax_acc.set_xlabel('Iteration')
    # ax_acc.set_xlim([0, len(los_lst)])
    ax_acc.set_ylabel('Step')
    # ax_acc.set_ylim([0, 100])
    ax_acc.tick_params(axis='y')
    ax_acc.grid(False)
    plt.legend()
    axins = ax_acc.inset_axes((0.3, 0.3, 0.5, 0.4))

    axins.scatter(x, step,c=acc,cmap=plt.cm.jet, s=2, label=f"Budget=200")

    mark_inset(ax_acc, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

    # 调整子坐标系的显示范围
    axins.set_xlim(0.5, 7.5)
    axins.set_ylim(20, 50)
    plt.savefig(os.path.join('figures',"Acc_vs_Budget.pdf"), dpi=800, format="pdf")
    plt.close(fig)

# Accuracy vs. number of denoising samples
def acc_vs_denoising_samples(log_path):
    with open(os.path.join(log_path, "config.yml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    df = pd.read_pickle(os.path.join(log_path, "df.pkl"))
    fig, ax = plt.subplots()
    linsp = np.linspace(1, config.purification.rand_smoothing_ensemble, num=config.purification.rand_smoothing_ensemble)
    correct_l = np.zeros_like(linsp)
    correct_s = np.zeros_like(linsp)
    correct_o = np.zeros_like(linsp)
    nSamples = np.zeros(1)
    for indices, rows in df.iterrows():
        for j in range(config.purification.rand_smoothing_ensemble):
            correct_l[j] += rows["pur_acc_list_l"][j]
            correct_s[j] += rows["pur_acc_list_s"][j]
            correct_o[j] += rows["pur_acc_list_o"][j]
        nSamples += rows["nData"]
    ax.plot(linsp, correct_l/nSamples*100., '-', color="r", label="acc_logit")
    ax.plot(linsp, correct_s/nSamples*100., '-', color="b", label="acc_softmax")
    ax.plot(linsp, correct_o/nSamples*100., '-', color="k", label="acc_onehot")
    plt.legend(loc='upper right', fontsize=18)    
    plt.rc('font', size=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)
    plt.savefig(os.path.join(log_path, "acc_denoising_iters.pdf"), dpi=800, format="pdf")
    plt.close(fig)

def acc_vs_denoising_samples_nat(log_path):
    with open(os.path.join(log_path, "config.yml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    df = pd.read_pickle(os.path.join(log_path, "df.pkl"))
    fig, ax = plt.subplots()
    linsp = np.linspace(1, config.purification.rand_smoothing_ensemble, num=config.purification.rand_smoothing_ensemble)
    correct_l = np.zeros_like(linsp)
    correct_s = np.zeros_like(linsp)
    correct_o = np.zeros_like(linsp)
    nSamples = np.zeros(1)
    for indices, rows in df.iterrows():
        for j in range(config.purification.rand_smoothing_ensemble):
            correct_l[j] += rows["nat_pur_acc_list_l"][j]
            correct_s[j] += rows["nat_pur_acc_list_s"][j]
            correct_o[j] += rows["nat_pur_acc_list_o"][j]
        nSamples += rows["nData"]
    ax.plot(linsp, correct_l/nSamples*100., '-', color="r", label="acc_logit")
    ax.plot(linsp, correct_s/nSamples*100., '-', color="b", label="acc_softmax")
    ax.plot(linsp, correct_o/nSamples*100., '-', color="k", label="acc_onehot")
    plt.legend(loc='upper right', fontsize=18)    
    plt.rc('font', size=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)
    plt.savefig(os.path.join(log_path, "nat_acc_denoising_iters.pdf"), dpi=800, format="pdf")
    plt.close(fig)

def acc_vs_denoising_samples_purens(log_path):
    with open(os.path.join(log_path, "config.yml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    df = pd.read_pickle(os.path.join(log_path, "df.pkl"))
    fig, ax = plt.subplots()
    linsp = np.linspace(1, config.purification.rand_smoothing_ensemble, num=config.purification.rand_smoothing_ensemble)
    correct_l = np.zeros_like(linsp)
    correct_s = np.zeros_like(linsp)
    correct_o = np.zeros_like(linsp)
    nSamples = np.zeros(1)
    for indices, rows in df.iterrows():
        for j in range(config.purification.rand_smoothing_ensemble):
            correct_l[j] += rows["purens_acc_list_l"][j]
            correct_s[j] += rows["purens_acc_list_s"][j]
            correct_o[j] += rows["purens_acc_list_o"][j]
        nSamples += rows["nData"]
    ax.plot(linsp, correct_l/nSamples*100., '-', color="r", label="acc_logit")
    ax.plot(linsp, correct_s/nSamples*100., '-', color="b", label="acc_softmax")
    ax.plot(linsp, correct_o/nSamples*100., '-', color="k", label="acc_onehot")
    plt.legend(loc='upper right', fontsize=18)    
    plt.rc('font', size=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)
    plt.savefig(os.path.join(log_path, "acc_denoising_iters_purens.pdf"), dpi=800, format="pdf")
    plt.close(fig)

def acc_vs_denoising_samples_purens_nat(log_path):
    with open(os.path.join(log_path, "config.yml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    df = pd.read_pickle(os.path.join(log_path, "df.pkl"))
    fig, ax = plt.subplots()
    linsp = np.linspace(1, config.purification.rand_smoothing_ensemble, num=config.purification.rand_smoothing_ensemble)
    correct_l = np.zeros_like(linsp)
    correct_s = np.zeros_like(linsp)
    correct_o = np.zeros_like(linsp)
    nSamples = np.zeros(1)
    for indices, rows in df.iterrows():
        for j in range(config.purification.rand_smoothing_ensemble):
            correct_l[j] += rows["nat_purens_acc_list_l"][j]
            correct_s[j] += rows["nat_purens_acc_list_s"][j]
            correct_o[j] += rows["nat_purens_acc_list_o"][j]
        nSamples += rows["nData"]
    ax.plot(linsp, correct_l/nSamples*100., '-', color="r", label="acc_logit")
    ax.plot(linsp, correct_s/nSamples*100., '-', color="b", label="acc_softmax")
    ax.plot(linsp, correct_o/nSamples*100., '-', color="k", label="acc_onehot")
    plt.legend(loc='upper right', fontsize=18)    
    plt.rc('font', size=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)
    plt.savefig(os.path.join(log_path, "nat_acc_denoising_iters_purens.pdf"), dpi=800, format="pdf")
    plt.close(fig)

def acc_vs_step_ensemble(log_path):
    with open(os.path.join(log_path, "config.yml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    df = pd.read_pickle(os.path.join(log_path, "df.pkl"))
    fig, ax = plt.subplots()
    linsp = np.linspace(1, config.purification.max_iter, config.purification.max_iter)
    correct_l = np.zeros_like(linsp)
    correct_s = np.zeros_like(linsp)
    correct_o = np.zeros_like(linsp)
    nSamples = np.zeros(1)
    for indices, rows in df.iterrows():
        for j in range(config.purification.max_iter):
            correct_l[j] += rows["purens_acc_list_all_l"][j]
            correct_s[j] += rows["purens_acc_list_all_s"][j]
            correct_o[j] += rows["purens_acc_list_all_o"][j]
    nSamples += rows["nData"]
    ax.plot(linsp, correct_l/nSamples*100., '-', color="r", label="acc_logit")
    ax.plot(linsp, correct_s/nSamples*100., '-', color="b", label="acc_softmax")
    ax.plot(linsp, correct_o/nSamples*100., '-', color="k", label="acc_onehot")
    plt.legend(loc='lower right', fontsize=18)    
    plt.rc('font', size=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)
    plt.xlabel('purification steps')
    plt.ylabel('accuracy (%)')
    plt.savefig(os.path.join(log_path, "acc_purstepsize.pdf"), dpi=800, format="pdf")
    plt.close(fig)
    
def acc_vs_step_ensemble_nat(log_path):
    with open(os.path.join(log_path, "config.yml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    df = pd.read_pickle(os.path.join(log_path, "df.pkl"))
    fig, ax = plt.subplots()
    linsp = np.linspace(1, config.purification.max_iter, config.purification.max_iter)
    correct_l = np.zeros_like(linsp)
    correct_s = np.zeros_like(linsp)
    correct_o = np.zeros_like(linsp)
    nSamples = np.zeros(1)
    for indices, rows in df.iterrows():
        for j in range(config.purification.max_iter):
            correct_l[j] += rows["nat_purens_acc_list_all_l"][j]
            correct_s[j] += rows["nat_purens_acc_list_all_s"][j]
            correct_o[j] += rows["nat_purens_acc_list_all_o"][j]
        nSamples += rows["nData"]
    ax.plot(linsp, correct_l/nSamples*100., '-', color="r", label="acc_logit")
    ax.plot(linsp, correct_s/nSamples*100., '-', color="b", label="acc_softmax")
    ax.plot(linsp, correct_o/nSamples*100., '-', color="k", label="acc_onehot")
    plt.legend(loc='upper right', fontsize=18)    
    plt.rc('font', size=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)
    plt.savefig(os.path.join(log_path, "nat_acc_purstepsize.pdf"), dpi=800, format="pdf")
    plt.close(fig)