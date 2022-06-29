from math import sqrt,floor
import pandas as pd 
from plot import *


def plot_acc_vs_iter(df,step_size):
    steps = df[(df["Step_size"]==step_size)&(df["Conditional guide"]==False)]
    steps = steps.sort_values(by=['Iteration'],ascending = [True])
    steps = steps.drop_duplicates(subset=['Iteration'], keep='last')
    acc_vs_iter(steps, step_size)

def plot_acc_vs_step(df,iter_num):
    iters = df[(df["Iteration"]==iter_num)&(df["Conditional guide"]==False) ]
    iters = iters.sort_values(by=['Step_size'],ascending = [True])
    iters = iters.drop_duplicates(subset=['Step_size'], keep='last')
    acc_vs_step(iters,iter_num)
    
def plot_acc_vs_budget_iter(df,num):
    iter_total = []
    for iter_num in range(num):
        iters = df[(df["Iteration"]==iter_num+1)&(df["Conditional guide"]==False) ]
        iters = iters.sort_values(by=['Step_size'],ascending = [True])
        iters = iters.drop_duplicates(subset=['Step_size'], keep='last')
        iter_total.append(iters)
    acc_vs_budget_iter(iter_total,num)

def plot_acc_vs_budget_iter_guide(df,num):
    iter_total = []
    for iter_num in range(num):
        iters = df[(df["Iteration"]==iter_num+1)&(df["Conditional guide"]== True)&(df['guide_scale']==f'{6e5}')&(df['Guide Mode']=='VAR') ]
        iters = iters.sort_values(by=['Step_size'],ascending = [True])
        iters = iters.drop_duplicates(subset=['Step_size'], keep='last')
        iter_total.append(iters)
    acc_vs_budget_iter_guide(iter_total,num)

def plot_acc_vs_budget_step(df,num):
    total = []
    for i in range(num):
        steps = df[(df["Step_size"]==i*10+10)&(df["Conditional guide"]==False)]
        steps = steps.sort_values(by=['Iteration'],ascending = [True])
        steps = steps.drop_duplicates(subset=['Iteration'], keep='last')
        total.append(steps)

    acc_vs_budget_step(total,num)

def plot_acc_vs_iters(df,steps_list):
    total = []
    for i in steps_list:
        steps = df[(df["Step_size"]==i)&(df["Conditional guide"]==False)]
        steps = steps.sort_values(by=['Iteration'],ascending = [True])
        steps = steps.drop_duplicates(subset=['Iteration'], keep='last')
        total.append(steps)
    acc_vs_iters(total, steps_list)


def plot_acc_vs_n(df,budget):
    df_total = []
    for n in range(len(budget)):
        step_list = []
        iter_list = []
        for i in range(floor(sqrt(budget[n]))):
            j = floor(budget[n]/(i+1))
            if (i+1) != j:
                step_list += [i+1,j]
                iter_list += [j,i+1]
            else:
                step_list += [i+1]
                iter_list += [j]
        step_list.sort()
        iter_list.sort(reverse=True)
        df_temp = df[(df["Iteration"]==iter_list[-1])&(df["Step_size"]==step_list[-1])&(df["Conditional guide"]==False) ]
        for i in range(len(step_list)-1):
            df_temp = df_temp.append(df[(df["Iteration"]==iter_list[i])&(df["Step_size"]==step_list[i])&(df["Conditional guide"]==False) ])
        df_temp = df_temp.sort_values(by=['Step_size'],ascending = [True])
        df_temp = df_temp.drop_duplicates(subset=['Step_size'], keep='last')
        df_total.append(df_temp)
    acc_vs_n(df_total,budget)

def plot_acc_vs_budget(df,budget):
    df_total = []
    # for n in range(len(budget)):
    df2 = pd.DataFrame(data=None, columns=df.columns, index=df.index)
    for i in range(1,budget):
        for j in range(1,budget):
            if i*j<budget: 
                df2 = df2.append(df[(df["Iteration"]==i)&(df["Step_size"]==j)&(df["Conditional guide"]==False) ])
    df2 = df2.drop_duplicates(subset=['Step_size','Iteration'], keep='last')
    df2 = df2.sort_values(by=['accurancy'],ascending = [False])

    acc_vs_budget(df2)


def plot_guide_compare(df,scale_list):
    # plor guided acc vs unguided acc
    df_list = []
    for i in range(len(scale_list)):
        df_temp = df[(df["Iteration"]==1)&(df["Conditional guide"]== True)&(df['guide_scale']==f'{scale_list[i]}')&(df['Guide Mode']=='VAR')]
        df_temp = df_temp.sort_values(by=['Step_size'],ascending = [True])
        df_temp = df_temp.drop_duplicates(subset=['Step_size'], keep='last')
        df_list.append(df_temp)

    df_temp = df[(df["Iteration"]==1)&(df["Conditional guide"]==False) ]
    df_temp = df_temp.sort_values(by=['Step_size'],ascending = [True])
    df_temp = df_temp.drop_duplicates(subset=['Step_size'], keep='last')
    df_list.append(df_temp)
    guide_compare_vs_step(scale_list,df_list)


def plot_var_vs_cons(df):
    df1 = df[(df["Iteration"]==1)&(df["Conditional guide"]==True)&(df["guide_scale"]=='50')&(df['Guide Mode']=='CONSTANT') ]
    df1 = df1.sort_values(by=['Step_size'],ascending = [True])
    df1 = df1.drop_duplicates(subset=['Step_size'], keep='last')
    df2 = df[(df["Iteration"]==1)&(df["Conditional guide"]== True)&(df['guide_scale']=='50')&(df['Guide Mode']=='VAR')]
    df2 = df2.sort_values(by=['Step_size'],ascending = [True])
    df2 = df2.drop_duplicates(subset=['Step_size'], keep='last')
    # print(df1)
    # print(df2)
    var_vs_cons(df1,df2)



if __name__ == '__main__': 
    df = pd.read_csv('./utils/result_forplot.csv',index_col=0)
    # plot_acc_vs_step(df,4)
    # plot_acc_vs_iter(df,50)
    # plot_guide_compare(df,[30,50,75,100])
    # plot_acc_vs_n(df,[100,150,200,250,300])
    # plot_acc_vs_budget_step(df,5)
    # plot_acc_vs_iters(df,3)
    # plot_acc_vs_budget_iter(df,5)
    # plot_guide_compare(df,[6e5])
    # plot_acc_vs_budget_iter_guide(df,4)
    # plot_acc_vs_budget(df,200)

    # plot_guide_compare(df,[6e5])
    plot_acc_vs_iters(df,[20,30,50])