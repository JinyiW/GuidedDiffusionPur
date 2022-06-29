# figure out how diffusion steps diverse
from distutils.command.config import config
from utils import *
from clf_models.networks import *
from attacks import *
from datetime import datetime
import tqdm
import pandas as pd
import torchvision
from purification.diff_purify import *
from pytorch_diffusion.diffusion import Diffusion
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


__all__ = ['Empirical']

class Empirical():
    def __init__(self,args,config):
        self.args = args 
        self.config = config 
        self.config.attack.if_attack = True

    def run(self, log_progress):
        # Normalize on classifiers 
        transform_raw_to_clf = raw_to_clf(self.config.structure.dataset)

        # Output log file configuration
        sys.stdout = log_progress
        # log_output = open(os.path.join(self.args.log, "log_output"), "w")

        # Import dataset
        start_time = datetime.now()
        print("[{}] Start importing dataset {}".format(str(datetime.now()), self.config.structure.dataset))
        if self.config.structure.dataset == 'CIFAR10-C':
            self.config.attack.if_attack = False
            testLoader_list = importData(dataset=self.config.structure.dataset, train=False, shuffle=False, bsize=self.config.structure.bsize)
            testLoader = testLoader_list[self.config.structure.CIFARC_CLASS-1][self.config.structure.CIFARC_SEV-1]
        elif self.config.structure.dataset == 'TinyImageNet':
            data_transforms = transforms.Compose([
                    transforms.ToTensor(),
            ])
            path_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
            path_root = os.path.join(path_root, "datasets")
            image_datasets = TINYIMAGENET(path_root,train=False,transform=data_transforms)
            testLoader = torch.utils.data.DataLoader(image_datasets, batch_size=self.config.structure.bsize, shuffle=True, num_workers=64)
        elif self.config.structure.dataset == 'ImageNet' and self.config.attack.attack_method == 'bpda_strong':
            testLoader = importData(dataset=self.config.structure.dataset, train=False, shuffle=True, bsize=self.config.structure.bsize)
        elif self.config.structure.dataset == 'ImageNet-C':
            self.config.attack.if_attack = False
            testLoader = importData(
                dataset=self.config.structure.dataset, train=False, shuffle=False, bsize=self.config.structure.bsize,
                distortion_name=self.config.structure.distortion_name, severity=self.config.structure.severity )
        # elif self.config.structure.dataset == 'ImageNet' and self.config.structure.run_samples <= 10000:
        #     testLoader = importData(dataset=self.config.structure.dataset, train=False, shuffle=True, bsize=self.config.structure.bsize)
        else:
            testLoader = importData(dataset=self.config.structure.dataset, train=False, shuffle=False, bsize=self.config.structure.bsize)
        print("[{}] Finished importing dataset {}".format(str(datetime.now()), self.config.structure.dataset))
        
        # Import classifier networks
        start_time = datetime.now()
        print("[{}] Start importing network".format(str(datetime.now())))
        if self.config.structure.dataset in ["ImageNet","ImageNet-C"]:
            if self.config.structure.classifier == 'ResNet152':
                network_clf = torchvision.models.resnet152(pretrained=True).to(self.config.device.clf_device)
            elif self.config.structure.classifier == 'ResNet50':
                network_clf = torchvision.models.resnet50(pretrained=True).to(self.config.device.clf_device)
            network_clf.eval()
        elif self.config.structure.clf_log not in ["cifar10_carmon", "cifar10_wu", "cifar10_zhang"]:
            network_clf = eval(self.config.structure.classifier)().to(self.config.device.clf_device)
            network_clf = torch.nn.DataParallel(network_clf)

        if self.config.structure.dataset in ["CIFAR10", "CIFAR10-C", "CIFAR100"]: # CIFAR10 setting, trained by WideResNet
            states_att = torch.load(os.path.join('clf_models/run/logs', self.config.structure.clf_log, '{}.t7'.format(self.config.classification.checkpoint)), map_location=self.config.device.clf_device) # Temporary t7 setting
            network_clf = states_att['net'].to(self.config.device.clf_device)
        # elif self.config.structure.dataset in ["ImageNet"]:   # Get network_clf from loaded network

        #import Diff Pretrained network
        if self.config.structure.dataset in ["CIFAR10", "CIFAR10-C"]:
            model_name = 'ema_cifar10'
            diffusion = Diffusion.from_pretrained(model_name, device=self.config.device.diff_device)
        elif self.config.structure.dataset in ["ImageNet","ImageNet-C"]:
            print("creating model and diffusion...")
            model, diffusion = create_model_and_diffusion(
                **args_to_dict(self.config.net, model_and_diffusion_defaults().keys())
            )
            model.load_state_dict(
                torch.load(self.config.net.model_path, map_location="cpu")
            )
            model.to(self.config.device.clf_device)
            if self.config.net.use_fp16:
                model.convert_to_fp16()
            model.eval() 

        df_columns = ["Epoch", "nData", "att_time", "pur_time", "clf_time", \
                        "std_acc", "att_acc", "pur_acc_l", "pur_acc_s", "pur_acc_o", \
                        "pur_acc_list_l", "pur_acc_list_s", "pur_acc_list_o","count_att" ,"count_diff"]
        if self.config.purification.purify_natural:
            df_columns.append("nat_pur_acc_l")
            df_columns.append("nat_pur_acc_s")
            df_columns.append("nat_pur_acc_o")
            df_columns.append("nat_pur_acc_list_l")
            df_columns.append("nat_pur_acc_list_s")
            df_columns.append("nat_pur_acc_list_o")
        df = pd.DataFrame(columns=df_columns)

        # Run
        for i, (x,y) in enumerate(tqdm.tqdm(testLoader)):
            if i<self.config.structure.start_epoch:
                continue
            if i>self.config.structure.end_epoch:
                break
            
            start_time = datetime.now()
            print("[{}] Epoch {}".format(str(datetime.now()), i))
            x = preprocess(x, self.config.structure.dataset) #preprocess cifar10-c
            x = x.to(self.config.device.diff_device)
            y = y.to(self.config.device.diff_device).long()

            ### ATTACK
            if self.config.attack.if_attack:
                if self.config.structure.dataset in ["CIFAR10", "CIFAR10-C"]:
                    x_adv, success, acc = eval(self.config.attack.attack_method)(x, y,diffusion, network_clf, self.config)
                elif self.config.structure.dataset in ["ImageNet"]:
                    x_adv, success, acc = eval(self.config.attack.attack_method)(x, y,diffusion, network_clf, self.config,model = model)
                attack_time = elapsed_seconds(start_time, datetime.now())
                print("[{}] Epoch {}:\t{:.2f} seconds to attack {} data".format(str(datetime.now()), i, attack_time, self.config.structure.bsize))
            else:
                x_adv = x
                attack_time = 0.0

            # out_path = os.path.join(self.args.log, f"attacked_samples_{i}.npz")
            # print(f"saving to {out_path}")
            # np.savez(out_path, x_adv.clone().detach().to("cpu").numpy())

            ### PURIFICATION
            x_pur_list_list = []
            start_time = datetime.now()
            print("[{}] Epoch {}:\tBegin purifying {} attacked images".format(str(datetime.now()), i, self.config.structure.bsize))
            for j in range(self.config.purification.path_number):
                if self.config.structure.dataset in ["CIFAR10", "CIFAR10-C"]:
                    x_pur_list = diff_purify(
                        x_adv, diffusion, 
                        self.config.purification.max_iter, 
                        mode="purification", 
                        config=self.config
                        )
                elif self.config.structure.dataset in ["ImageNet","ImageNet-C"]:
                    x_pur_list = purify_imagenet(x_adv, diffusion, model, 
                        self.config.purification.max_iter, 
                        mode="purification", 
                        config=self.config)
                x_pur_list_list.append(x_pur_list)
            purify_attacked_time = elapsed_seconds(start_time, datetime.now())
            print("[{}] Epoch {}:\t{:.2f} seconds to purify {} attacked images".format(str(datetime.now()), i, purify_attacked_time, self.config.structure.bsize))
            
            # purify natural image
            if self.config.purification.purify_natural:
                x_nat_pur_list_list = []
                start_time = datetime.now()
                print("[{}] Epoch {}:\tBegin purifying {} natural images".format(str(datetime.now()), i, self.config.structure.bsize))
                for j in range(self.config.purification.path_number):
                    if self.config.structure.dataset in ["CIFAR10", "CIFAR10-C"]:
                        x_nat_pur_list = diff_purify(
                            x, diffusion, 
                            self.config.purification.max_iter, 
                            mode="purification", 
                            config=self.config
                            )
                    elif self.config.structure.dataset in ["ImageNet"]:
                        x_nat_pur_list = purify_imagenet(x, diffusion, model, 
                            self.config.purification.max_iter, 
                            mode="purification", 
                            config=self.config)
                    x_nat_pur_list_list.append(x_nat_pur_list)
                purify_natural_time = elapsed_seconds(start_time, datetime.now())
                print("[{}] Epoch {}:\t{:.2f} seconds to purify {} natural images".format(str(datetime.now()), i, purify_natural_time, self.config.structure.bsize))

            ### CLASSIFICATION: logit/softmax/onehot
            # Classify natural and attacked images
            with torch.no_grad():
                y_t = network_clf(transform_raw_to_clf(x.clone().detach()).to(self.config.device.clf_device))
                y_adv_t = network_clf(transform_raw_to_clf(x_adv.clone().detach()).to(self.config.device.clf_device))
                nat_correct = torch.eq(torch.argmax(y_t, dim=1), y.clone().to(self.config.device.clf_device)).float().sum()
                att_correct = torch.eq(torch.argmax(y_adv_t, dim=1), y.clone().to(self.config.device.clf_device)).float().sum()
                att_label = torch.argmax(y_adv_t, dim=1).to('cpu').numpy()

            # Classify all purified attacked images
            with torch.no_grad():
                start_time = datetime.now()
                print("[{}] Epoch {}:\tBegin predicting {} purified attacked images".format(str(datetime.now()), i, self.config.structure.bsize))
                att_list_list_dict = gen_ll(x_pur_list_list, network_clf, transform_raw_to_clf, self.config)
                classify_attacked_time = elapsed_seconds(start_time, datetime.now())
                print("[{}] Epoch {}:\t{:.2f} seconds to predict {} purified attacked images".format(str(datetime.now()), i, classify_attacked_time, self.config.structure.bsize))

                # Classify all purified natural images
                if self.config.purification.purify_natural:
                    start_time = datetime.now()
                    print("[{}] Epoch {}:\tBegin predicting {} purified natural images".format(str(datetime.now()), i, self.config.structure.bsize))
                    nat_list_list_dict = gen_ll(x_nat_pur_list_list, network_clf, transform_raw_to_clf, self.config)
                    classify_natural_time = elapsed_seconds(start_time, datetime.now())
                    print("[{}] Epoch {}:\t{:.2f} seconds to predict {} purified natural images".format(str(datetime.now()), i, classify_natural_time, self.config.structure.bsize))

            ### PERFORMANCE ANALYSIS
            att_acc, att_acc_iter, cls_label = acc_final_step(att_list_list_dict, y)
            if self.config.purification.purify_natural:
                nat_acc, nat_acc_iter,cls_label = acc_final_step(nat_list_list_dict, y)

            # count the misclassification from PGD or diffusion progress
            count_att = 0
            count_diff = 0
            for j in range(len(cls_label)):
                if cls_label[j] != y[j]:
                    if cls_label[j] == att_label[j]:
                        count_att +=1 
                    else:
                        count_diff +=1 

            new_row = \
                    {
                        "Epoch": i+1,
                        "nData": self.config.structure.bsize,
                        "att_time": attack_time,
                        "pur_time": purify_attacked_time,
                        "clf_time": classify_attacked_time,
                        "std_acc": nat_correct.to('cpu').numpy(),
                        "att_acc": att_correct.to('cpu').numpy(),
                        "pur_acc_l": att_acc["logit"],
                        "pur_acc_s": att_acc["softmax"],
                        "pur_acc_o": att_acc["onehot"],
                        "pur_acc_list_l": att_acc_iter["logit"],
                        "pur_acc_list_s": att_acc_iter["softmax"],
                        "pur_acc_list_o": att_acc_iter["onehot"],
                        "count_att": count_att,
                        "count_diff": count_diff
                    }
            if self.config.purification.purify_natural:
                new_row["nat_pur_acc_l"] = nat_acc["logit"]
                new_row["nat_pur_acc_s"] = nat_acc["softmax"]
                new_row["nat_pur_acc_o"] = nat_acc["onehot"]
                new_row["nat_pur_acc_list_l"] = nat_acc_iter["logit"]
                new_row["nat_pur_acc_list_s"] = nat_acc_iter["softmax"]
                new_row["nat_pur_acc_list_o"] = nat_acc_iter["onehot"]
            
            df = df.append(new_row, ignore_index=True)
        df.to_csv(os.path.join(self.args.log, f"result_{self.config.device.rank}.csv"))
        