import os
import argparse
import copy
import yaml
import json
import pickle

from tqdm import tqdm
import numpy as np

from extorch.utils import run_processes, set_seed


class WrapperArch(object):
    def __init__(self, arch, key, all_ranks):
        self.arch = arch
        self.key = key
        self.all_ranks = [ranks[self.arch][self.key] for ranks in all_ranks]

    def __lt__(self, b):
        return np.mean(self.all_ranks) < np.mean(b.all_ranks)
    
    def __gt__(self, b):
        return np.mean(self.all_ranks) > np.mean(b.all_ranks)


if __name__ == "__main__":
    default_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    default_train_data = os.path.join(default_parent_dir, "data", "CVPR_2022_NAS_Track2_train.json")
    default_test_data = os.path.join(default_parent_dir, "data", "CVPR_2022_NAS_Track2_test.json")
    default_cfg_path = os.path.join(default_parent_dir, "configuration")

    parser = argparse.ArgumentParser(prog = "main.py")

    parser.add_argument("cfg-path", type = str, default = default_cfg_path)
    parser.add_argument("--split-num", type = int, default = 7)
    parser.add_argument("--gpu", type = int, default = 0, help = "gpu device id")
    parser.add_argument("--num-workers", default = 4, type = int)
    parser.add_argument("--report_freq", default = 200, type = int)
    parser.add_argument("--seed", default = 100, type = int)
    parser.add_argument("--train-dir", default = None, help = "Save train log/results into TRAIN_DIR")
    parser.add_argument("--save-every", default = 100, type = int)
    parser.add_argument("--train-data", default = default_train_data, help = "Training Datasets data")
    parser.add_argument("--test-data", default = default_test_data, help = "Evaluate Datasets data")
    
    args = parser.parse_args()

    set_seed(args.seed)

    task_names = [
        "cplfw_rank",  # individual
        "veriwild_rank", # individual 
        "msmt17_rank", # individual
        "sop_rank",  # individual
        "vehicleid_rank", # individual
        "veriwild_rank_for_dukemtmc",
        "market1501_rank", # individual
        "dukemtmc_rank",  # finetune from veriwild
        "veri_rank" # individual
    ]


    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)

    # Split Data
    with open(args.train_data, "r") as rf:
        train_data = json.load(rf)

    data_path = os.path.join(args.train_dir, "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    arch_lst = list(train_data.keys())
    all_index = np.arange(len(arch_lst))
    np.random.shuffle(all_index)
    split_index = np.array_split(all_index, args.split_num)

    train_data_path_lst = []
    valid_data_path_lst = []

    for i in range(args.split_num):
        split_data_path = os.path.join(data_path, "{}".format(i))
        if not os.path.exists(split_data_path):
            os.makedirs(split_data_path)

        # Split train data
        train_data_path = os.path.join(split_data_path, "split_train.json")
        train_data_path_lst.append(train_data_path)

        all_train_index = []
        for j in range(args.split_num):
            if j != i:
                all_train_index += list(split_index[j])

        split_train_data = {
            arch_lst[all_index[ind]]: train_data[arch_lst[all_index[ind]]]
            for ind in all_train_index
        }

        with open(train_data_path, "w") as wf:
            json.dump(split_train_data, wf)

        # Split valid data
        valid_data_path = os.path.join(split_data_path, "split_valid.json")
        valid_data_path_lst.append(valid_data_path)

        split_valid_data = {
            arch_lst[all_index[ind]]: train_data[arch_lst[all_index[ind]]]
            for ind in split_index[i]
        }

        with open(valid_data_path, "w") as wf:
            json.dump(split_valid_data, wf)

    # Generate commands
    commands = []
    for i, task in enumerate(task_names):

        # Load configuration
        cfg_file = os.path.join(default_cfg_path, "{}.yaml".format(task))
        
        task_path = os.path.join(args.train_dir, task)
        if not os.path.exists(task_path):
            os.makedirs(task_path)

        for j in range(args.split_num):
            final_path = os.path.join(task_path, "{}".format(j))

            if task in ["cplfw_rank", "veriwild_rank", "msmt17_rank", "sop_rank", "vehicleid_rank", "veriwild_rank_for_dukemtmc", "market1501_rank", "veri_rank"]:
                command = "python {} {} --gpu {} --num-workers {} --report_freq {} --seed {} --train-dir {} --train-data {} --valid-data {} --test-data {}".format(
                        os.path.join(default_parent_dir, "train_individual.py"), 
                        cfg_file, 
                        args.gpu, 
                        args.num_workers, 
                        args.report_freq, 
                        args.seed + j, 
                        final_path, 
                        train_data_path_lst[j], 
                        valid_data_path_lst[j],
                        args.test_data
                )
            
            elif task in ["dukemtmc_rank"]:
                command = "python {} {} --gpu {} --num-workers {} --report_freq {} --seed {} --train-dir {} --train-data {} --valid-data {} --test-data {} --load {}".format(
                        os.path.join(default_parent_dir, "train_individual.py"), 
                        cfg_file, 
                        args.gpu, 
                        args.num_workers, 
                        args.report_freq, 
                        args.seed + j, 
                        final_path, 
                        train_data_path_lst[j], 
                        valid_data_path_lst[j],
                        args.test_data,
                        os.path.join(os.path.join(args.train_dir, "veriwild_rank_for_dukemtmc", str(j), "best.ckpt"))
                )
            
            commands.append(command)

    failed_commands = run_processes(commands)

    task_names.remove("veriwild_rank_for_dukemtmc")

    if len(failed_commands) > 0:
        print("Number of failed commands: {}".format(len(failed_commands)))
        import ipdb
        ipdb.set_trace()

    else:
        with open(args.test_data, "r") as rf:
            test_data = json.load(rf)
        back_test_data = copy.deepcopy(test_data)

        for task in tqdm(task_names):
            task_path = os.path.join(args.train_dir, task)
            all_ranks = []
            
            for j in range(args.split_num):
                final_path = os.path.join(task_path, "{}".format(j))
                results_path = os.path.join(final_path, "results_rank.pkl")
                
                with open(results_path, "rb") as rf:
                    ranks = pickle.load(rf)
                
                translate_test_rank = copy.deepcopy(back_test_data)
                for key, rank in zip(list(test_data.keys()), ranks):
                    translate_test_rank[key][task] = int(rank)
                
                all_ranks.append(translate_test_rank)

            all_archs = [WrapperArch(arch, task, all_ranks) for arch in list(test_data.keys())]
            all_archs = np.sort(all_archs)

            for j, arch in enumerate(all_archs):
                test_data[arch.arch][task] = int(j)

        with open(os.path.join(args.train_dir, "CVPR_2022_NAS_Track2_submit_A.json"), "w") as wf:
            json.dump(test_data, wf)
