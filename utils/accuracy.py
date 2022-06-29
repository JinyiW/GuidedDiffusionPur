import torch.nn.functional as F
import torch
import numpy as np

def gen_ll(x_ll, network_clf, transform_raw_to_clf, config):
    # z: list of lists that consist of logits
    logit_ll = []
    softmax_ll = []
    onehot_ll = []
    for i in range(len(x_ll)):
        chain_length = len(x_ll[i])
        logit_l = []
        softmax_l = []
        onehot_l = []
        if config.classification.classify_all_steps:
            for j in range(chain_length):
                x_t = x_ll[i][j].clone().detach()
                logit_l.append(network_clf(transform_raw_to_clf(x_t)))
                softmax_l.append(F.softmax(logit_l[j], dim=1))
                onehot_l.append(torch.argmax(softmax_l[j], dim=1))
        else:
            x_t = x_ll[i][-1].clone().detach()
            logit_l.append(network_clf(transform_raw_to_clf(x_t)))
            softmax_l.append(F.softmax(logit_l[0], dim=1))
            onehot_l.append(torch.argmax(softmax_l[0], dim=1))
        logit_ll.append(logit_l)
        softmax_ll.append(softmax_l)
        onehot_ll.append(onehot_l)
    return {"logit": logit_ll, "softmax": softmax_ll, "onehot": onehot_ll}

def acc_all_step(truth_ll, ground_label, config):
    # accuracy, involving all step
    total_logit = torch.zeros_like(truth_ll["logit"][0][0])
    total_softmax = torch.zeros_like(truth_ll["softmax"][0][0])
    total_onehot = torch.zeros_like(truth_ll["softmax"][0][0]) # NOT typo, "softmax" is 
    # asmpytotics of accuracy by increasing noise injected samples
    list_noisy_inputs_logit = [] # [1 noisy input, 2 noisy inputs, 3 noisy inputs, ...]
    list_noisy_inputs_softmax = []
    list_noisy_inputs_onehot = []
    for i in range(len(truth_ll["logit"])): # list of list of [bsize, nClass]
        for j in range(len(truth_ll["logit"][i])):
            total_logit += truth_ll["logit"][i][j] / len(truth_ll["logit"][i])
        list_noisy_inputs_logit.append(torch.eq(torch.argmax(total_logit, dim=1), ground_label).sum().float().to('cpu').numpy())
    for i in range(len(truth_ll["softmax"])): # list of list of [bsize, nClass]
        for j in range(len(truth_ll["softmax"][i])):
            total_softmax += truth_ll["softmax"][i][j] / len(truth_ll["softmax"][i])
        list_noisy_inputs_softmax.append(torch.eq(torch.argmax(total_softmax, dim=1), ground_label).sum().float().to('cpu').numpy())
    for i in range(len(truth_ll["onehot"])): # list of list of [bsize]
        for j in range(len(truth_ll["onehot"][i])):
            for k in range(truth_ll["onehot"][i][j].shape[0]):
                total_onehot[k, truth_ll["onehot"][i][j][k]] += 1./len(truth_ll["onehot"][i])
        list_noisy_inputs_onehot.append(torch.eq(torch.argmax(total_onehot, dim=1), ground_label).sum().float().to('cpu').numpy())
    max_purens = config.purification.max_iter
    # asmpytotics of accuracy by noise ensemble
    list_pur_steps_logit = [] # [first 1 steps, [1:2] steps, [1:3] steps, ...]
    list_pur_steps_softmax = []
    list_pur_steps_onehot = []
    list_each_step_logit = [] # [first 1 step, 2 step, 3 step, ...]
    list_each_step_softmax = []
    list_each_step_onehot = []
    total_logit_last_step = torch.zeros_like(total_logit) # [Last step]
    total_softmax_last_step = torch.zeros_like(total_softmax)
    total_onehot_last_step = torch.zeros_like(total_softmax)
    for j in range(max_purens):
        total_logit_each_step = torch.zeros_like(total_logit)
        total_softmax_each_step = torch.zeros_like(total_softmax)
        total_onehot_each_step = torch.zeros_like(total_onehot)
        for i in range(len(truth_ll["logit"])):
            total_logit_last_step += truth_ll["logit"][i][min(j, len(truth_ll["logit"][i])-1)]
            total_logit_each_step += truth_ll["logit"][i][min(j, len(truth_ll["logit"][i])-1)]
        list_pur_steps_logit.append(torch.eq(torch.argmax(total_logit_last_step, dim=1), ground_label).sum().float().to('cpu').numpy())
        list_each_step_logit.append(torch.eq(torch.argmax(total_logit_each_step, dim=1), ground_label).sum().float().to('cpu').numpy())
        for i in range(len(truth_ll["softmax"])):
            total_softmax_last_step += truth_ll["softmax"][i][min(j, len(truth_ll["softmax"][i])-1)]
            total_softmax_each_step += truth_ll["softmax"][i][min(j, len(truth_ll["softmax"][i])-1)]
        list_pur_steps_softmax.append(torch.eq(torch.argmax(total_softmax_last_step, dim=1), ground_label).sum().float().to('cpu').numpy())
        list_each_step_softmax.append(torch.eq(torch.argmax(total_softmax_each_step, dim=1), ground_label).sum().float().to('cpu').numpy())
        for i in range(len(truth_ll["onehot"])):
            for k in range(truth_ll["onehot"][i][min(j, len(truth_ll["onehot"][i])-1)].shape[0]):
                total_onehot_last_step[k, truth_ll["onehot"][i][min(j, len(truth_ll["onehot"][i])-1)][k]] += 1.
                total_onehot_each_step[k, truth_ll["onehot"][i][min(j, len(truth_ll["onehot"][i])-1)][k]] += 1.
        list_pur_steps_onehot.append(torch.eq(torch.argmax(total_onehot_last_step, dim=1), ground_label).sum().float().to('cpu').numpy())
        list_each_step_onehot.append(torch.eq(torch.argmax(total_onehot_each_step, dim=1), ground_label).sum().float().to('cpu').numpy())
    logit_correct = torch.eq(torch.argmax(total_logit, dim=1), ground_label).sum().float().to('cpu').numpy()
    softmax_correct = torch.eq(torch.argmax(total_softmax, dim=1), ground_label).sum().float().to('cpu').numpy()
    onehot_correct = torch.eq(torch.argmax(total_onehot, dim=1), ground_label).sum().float().to('cpu').numpy()
    list_noisy_inputs_logit = np.array(list_noisy_inputs_logit)
    list_noisy_inputs_softmax = np.array(list_noisy_inputs_softmax)
    list_noisy_inputs_onehot = np.array(list_noisy_inputs_onehot)
    list_pur_steps_logit = np.array(list_pur_steps_logit)
    list_pur_steps_softmax = np.array(list_pur_steps_softmax)
    list_pur_steps_onehot = np.array(list_pur_steps_onehot)

    return {"logit": logit_correct, "softmax": softmax_correct, "onehot": onehot_correct}, \
           {"logit": list_noisy_inputs_logit, "softmax": list_noisy_inputs_softmax, "onehot": list_noisy_inputs_onehot}, \
           {"logit": list_pur_steps_logit, "softmax": list_pur_steps_softmax, "onehot": list_pur_steps_onehot}, \
           {"logit": list_each_step_logit, "softmax": list_each_step_softmax, "onehot": list_each_step_onehot}

def acc_final_step(truth_ll, ground_label):
    # output: final_correct, correct
    total_logit = torch.zeros_like(truth_ll["logit"][0][0])
    total_softmax = torch.zeros_like(truth_ll["softmax"][0][0])
    total_onehot = torch.zeros_like(truth_ll["softmax"][0][0]) # NOT typo, "softmax" is right
    list_noisy_inputs_logit = []
    list_noisy_inputs_softmax = []
    list_noisy_inputs_onehot = []
    # accuracy, involving final step
    for i in range(len(truth_ll["logit"])): # list of list of [bsize, nClass]
        total_logit += truth_ll["logit"][i][-1]
        list_noisy_inputs_logit.append(torch.eq(torch.argmax(total_logit, dim=1), ground_label).sum().float().to('cpu').numpy())
    for i in range(len(truth_ll["softmax"])): # list of list of [bsize, nClass]
        total_softmax += truth_ll["softmax"][i][-1]
        list_noisy_inputs_softmax.append(torch.eq(torch.argmax(total_softmax, dim=1), ground_label).sum().float().to('cpu').numpy())
    for i in range(len(truth_ll["onehot"])): # list of list of [bsize]
        for k in range(truth_ll["onehot"][i][0].shape[0]):
            total_onehot[k][truth_ll["onehot"][i][-1][k]] += 1.
        list_noisy_inputs_onehot.append(torch.eq(torch.argmax(total_onehot, dim=1), ground_label).sum().float().to('cpu').numpy())
    logit_correct = torch.eq(torch.argmax(total_logit, dim=1), ground_label).sum().float().to('cpu').numpy()
    softmax_correct = torch.eq(torch.argmax(total_softmax, dim=1), ground_label).sum().float().to('cpu').numpy()
    onehot_correct = torch.eq(torch.argmax(total_onehot, dim=1), ground_label).sum().float().to('cpu').numpy()
    list_noisy_inputs_logit = np.array(list_noisy_inputs_logit)
    list_noisy_inputs_softmax = np.array(list_noisy_inputs_softmax)
    list_noisy_inputs_onehot = np.array(list_noisy_inputs_onehot)
    logit_result = torch.argmax(total_logit, dim=1).to('cpu').numpy()
    return {"logit": logit_correct, "softmax": softmax_correct, "onehot": onehot_correct}, \
           {"logit": list_noisy_inputs_logit, "softmax": list_noisy_inputs_softmax, "onehot": list_noisy_inputs_onehot}, logit_result