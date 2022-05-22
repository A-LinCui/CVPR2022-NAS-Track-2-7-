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
    default_test_data = os.path.join(default_parent_dir, "data", "CVPR_2022_NAS_Track2_test.json")

    parser = argparse.ArgumentParser(prog = "reproduce_from_ckpt.py")
    parser.add_argument("--split-num", type = int, default = 7)
    parser.add_argument("--train-dir", default = None, help = "Save train log/results into TRAIN_DIR")
    parser.add_argument("--test-data", default = default_test_data, help = "Evaluate Datasets data")
    args = parser.parse_args()

    task_names = [
        "cplfw_rank",  # individual
        "veriwild_rank", # individual 
        "msmt17_rank", # individual
        "sop_rank",  # individual
        "vehicleid_rank", # individual
        "market1501_rank", # distill from msmt
        "dukemtmc_rank",  # finetune from veriwild
        "veri_rank" # individual
    ]
    
    if True:
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
