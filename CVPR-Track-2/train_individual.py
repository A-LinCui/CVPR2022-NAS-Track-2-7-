# -*- coding: utf-8 -*-
# pylint: disable-all

import os
import sys
import shutil
import logging
import argparse
import random
import pickle
import json
import copy
import yaml

from scipy.stats import stats
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import setproctitle
from torch.utils.data import Dataset, DataLoader

from aw_nas import utils
from aw_nas.evaluator.arch_network import ArchNetwork
from aw_nas.common import get_search_space


class ViTDataset(Dataset):
    def __init__(self, data, specific_task = None):
        self.data = data
        self._len = len(self.data)
        self._keys = list(self.data.keys())
        self.specific_task = specific_task

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        data = self.data[self._keys[idx]]
        arch = data["arch"]
        if self.specific_task:
            rank = data[self.specific_task]
            return arch, rank


def train(train_loader, model, epoch, args, arch_network_type):
    objs = utils.AverageMeter()
    n_diff_pairs_meter = utils.AverageMeter()
    model.train()
    for step, (archs, ranks) in enumerate(train_loader):
        archs = np.array(archs)
        ranks = np.array(ranks)
        n = len(archs)
        
        # Compare data
        n_max_pairs = int(args.max_compare_ratio * n)
        rank_diff = np.array(ranks)[:, None] - np.array(ranks)
        rank_abs_diff_matrix = np.triu(np.abs(rank_diff), 1)
        ex_thresh_inds = np.where(rank_abs_diff_matrix > args.compare_threshold)
        ex_thresh_num = len(ex_thresh_inds[0])
        if ex_thresh_num > n_max_pairs:
            if args.choose_pair_criterion == "diff":
                keep_inds = np.argpartition(rank_abs_diff_matrix[ex_thresh_inds], -n_max_pairs)[-n_max_pairs:]
            elif args.choose_pair_criterion == "random":
                keep_inds = np.random.choice(np.arange(ex_thresh_num), n_max_pairs, replace=False)
            ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])
        archs_1, archs_2, better_lst = archs[ex_thresh_inds[1]], archs[ex_thresh_inds[0]], (rank_diff > 0)[ex_thresh_inds]
        
        n_diff_pairs = len(better_lst)
        n_diff_pairs_meter.update(float(n_diff_pairs))
        loss = model.update_compare(archs_1, archs_2, better_lst)
        objs.update(loss, n_diff_pairs)
        
        if step % args.report_freq == 0:
            n_pair_per_batch = (args.batch_size * (args.batch_size - 1)) // 2
            logging.info("train {:03d} [{:03d}/{:03d}] {:.4f}; {}".format(
                epoch, step, len(train_loader), objs.avg,
                "different pair ratio: {:.3f} ({:.1f}/{:3d})".format(
                    n_diff_pairs_meter.avg / n_pair_per_batch,
                    n_diff_pairs_meter.avg, n_pair_per_batch) if args.compare else ""))
    return objs.avg


def test_xp(true_scores, predict_scores):
    true_inds = np.argsort(true_scores)[::-1]
    true_scores = np.array(true_scores)
    reorder_true_scores = true_scores[true_inds]
    predict_scores = np.array(predict_scores)
    reorder_predict_scores = predict_scores[true_inds]
    ranks = np.argsort(reorder_predict_scores)[::-1]
    num_archs = len(ranks)
    # calculate precision at each point
    cur_inds = np.zeros(num_archs)
    passed_set = set()
    for i_rank, rank in enumerate(ranks):
        cur_inds[i_rank] = (cur_inds[i_rank - 1] if i_rank > 0 else 0) + \
                           int(i_rank in passed_set) + int(rank <= i_rank)
        passed_set.add(rank)
    patks = cur_inds / (np.arange(num_archs) + 1)
    THRESH = 100
    p_corrs = []
    for prec in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        k = np.where(patks[THRESH:] >= prec)[0][0] + THRESH
        arch_inds = ranks[:k][ranks[:k] < k]
        # stats.kendalltau(arch_inds, np.arange(len(arch_inds)))
        p_corrs.append((k, float(k)/num_archs, len(arch_inds), prec, stats.kendalltau(
            reorder_true_scores[arch_inds],
            reorder_predict_scores[arch_inds]).correlation))
    return p_corrs


def test_xk(true_scores, predict_scores):
    true_inds = np.argsort(true_scores)[::-1]
    true_scores = np.array(true_scores)
    reorder_true_scores = true_scores[true_inds]
    predict_scores = np.array(predict_scores)
    reorder_predict_scores = predict_scores[true_inds]
    ranks = np.argsort(reorder_predict_scores)[::-1]
    num_archs = len(ranks)
    patks = []
    for ratio in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        k = int(num_archs * ratio)
        p = len(np.where(ranks[:k] < k)[0]) / float(k)
        arch_inds = ranks[:k][ranks[:k] < k]
        patks.append((k, ratio, len(arch_inds), p, stats.kendalltau(
            reorder_true_scores[arch_inds],
            reorder_predict_scores[arch_inds]).correlation))
    return patks


def valid(val_loader, model, args, epoch, funcs=[]):
    model.eval()
    all_scores = []
    true_ranks = []
    for step, (archs, ranks) in enumerate(val_loader):
        scores = list(model.predict(archs).cpu().data.numpy())
        all_scores += scores
        true_ranks += list(ranks)

    if args.save_predict is not None:
        with open(args.save_predict, "wb") as wf:
            pickle.dump((true_ranks, all_scores), wf)

    corr = stats.kendalltau(true_ranks, all_scores).correlation
    funcs_res = [func(true_ranks, all_scores) for func in funcs]

    if args.train_dir is not None and epoch == args.epochs:
        index = np.argsort(true_ranks)
        plt.plot(np.array(true_ranks)[index], np.array(all_scores)[index])
        plt.savefig(os.path.join(args.train_dir, "result.png"), dpi = 600)

    return corr, funcs_res


def test(test_loader, model):
    model.eval()
    all_scores = []
    for step, (archs, _) in enumerate(test_loader):
        scores = list(model.predict(archs).cpu().data.numpy())
        all_scores += scores
    
    all_scores = np.array(all_scores)
    ranks = np.arange(0, len(all_scores))
    seq_ranks = np.argsort(all_scores)
    for i, idx in enumerate(seq_ranks):
        ranks[idx] = i
    return list(ranks)


def main(argv):
    parser = argparse.ArgumentParser(prog = "train_individual.py")
    parser.add_argument("cfg_file")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--report_freq", default=200, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--train-dir", default=None, help="Save train log/results into TRAIN_DIR")
    parser.add_argument("--save-every", default=50, type=int)
    parser.add_argument("--test-only", default=False, action="store_true")
    parser.add_argument("--test-funcs", default=None, help="comma-separated list of test funcs")
    parser.add_argument("--load", default=None, help="Load comparator from disk.")
    parser.add_argument("--eval-only-last", default=None, type=int,
                        help=("for pairwise compartor, the evaluation is slow,"
                              " only evaluate in the final epochs"))
    parser.add_argument("--save-predict", default=None, help="Save the predict scores")
    parser.add_argument("--train-data", default="data/CVPR_2022_NAS_Track2_train.json", help="Training Datasets data")
    parser.add_argument("--valid-data", required = True, type = str, help = "Validation Dataset Data")
    parser.add_argument("--test-data", default="data/CVPR_2022_NAS_Track2_test.json", help="Evaluate Datasets data")
    args = parser.parse_args(argv)

    setproctitle.setproctitle("python {} config: {}; train_dir: {}; cwd: {}"\
                              .format(__file__, args.cfg_file, args.train_dir, os.getcwd()))

    # log
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream = sys.stdout, level = logging.INFO,
                        format = log_format, datefmt = "%m/%d %I:%M:%S %p")

    if not args.test_only:
        assert args.train_dir is not None, "Must specificy `--train-dir` when training"
        # if training, setting up log file, backup config file
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        log_file = os.path.join(args.train_dir, "train.log")
        logging.getLogger().addFile(log_file)

        # copy config file
        backup_cfg_file = os.path.join(args.train_dir, "config.yaml")
        shutil.copyfile(args.cfg_file, backup_cfg_file)
    else:
        backup_cfg_file = args.cfg_file

    # cuda
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        logging.info("GPU device = %d" % args.gpu)
    else:
        logging.info("no GPU available, use CPU!!")

    if args.seed is not None:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    search_space = get_search_space("vit")

    logging.info("Loading data from {}; {}; {}".format(
        args.train_data, args.valid_data, args.test_data))
    
    with open(args.train_data, "r") as rf:
        train_data = json.load(rf)
    
    with open(args.valid_data, "r") as rf:
        valid_data = json.load(rf)
    
    with open(args.test_data, "r") as rf:
        test_data = json.load(rf)

    with open(backup_cfg_file, "r") as cfg_f:
        cfg = yaml.load(cfg_f)

    logging.info("Config: %s", cfg)
    arch_network_type = cfg.get("arch_network_type", "pointwise_comparator")
    model_cls = ArchNetwork.get_class_(arch_network_type)
    model = model_cls(search_space, **cfg.pop("arch_network_cfg"))
    if args.load is not None:
        logging.info("Load %s from %s", arch_network_type, args.load)
        model.load(args.load)
    model.to(device)

    args.__dict__.update(cfg)
    logging.info("Combined args: %s", args)
    
    # Split the training dataset
    logging.info("Number of architectures: train: %d; valid: %d; test: %d", \
            len(train_data), len(valid_data), len(test_data))

    train_data = ViTDataset(train_data, args.specific_task)
    valid_data = ViTDataset(valid_data, args.specific_task)
    test_data = ViTDataset(test_data, args.specific_task)

    train_loader = DataLoader(
        train_data, batch_size = args.batch_size, shuffle = True, pin_memory = True, num_workers = args.num_workers,
        collate_fn = lambda items: list(zip(*items)))
    val_loader = DataLoader(
        valid_data, batch_size = args.batch_size, shuffle = False, pin_memory = True, num_workers = args.num_workers,
        collate_fn = lambda items: list(zip(*items)))
    test_loader = DataLoader(
        test_data, batch_size = args.batch_size, shuffle = False, pin_memory = True, num_workers = args.num_workers,
        collate_fn = lambda items: list(zip(*items)))

    """
    # init test
    if not arch_network_type in {"pairwise_comparator", "random_forest"} or args.test_only:
        if args.test_funcs is not None:
            test_func_names = args.test_funcs.split(",")
        corr, func_res = valid(val_loader, model, args,
                               funcs=[globals()[func_name] for func_name in test_func_names]
                               if args.test_funcs is not None else [])
        if args.test_funcs is None:
            logging.info("INIT: kendall tau {:.4f}".format(corr))
        else:
            logging.info("INIT: kendall tau {:.4f};\n\t{}".format(
                corr,
                "\n\t".join(["{}: {}".format(name, res)
                             for name, res in zip(test_func_names, func_res)])))
    """

    if args.test_only:
        return

    best_corr = 0.
    
    for i_epoch in range(1, args.epochs + 1):
        model.on_epoch_start(i_epoch)
        avg_loss = train(train_loader, model, i_epoch, args, arch_network_type)
        logging.info("Train: Epoch {:3d}: train loss {:.4f}".format(i_epoch, avg_loss))
        if args.eval_only_last is None or (args.epochs - i_epoch < args.eval_only_last):
            train_corr, _ = valid(train_loader, model, args, i_epoch)
            corr, _ = valid(val_loader, model, args, i_epoch)
            logging.info("Valid: Epoch {:3d}: kendall tau {:.4f} / {:.4f}".format(i_epoch, train_corr, corr))
            if corr > best_corr:
                best_corr = corr
                save_path = os.path.join(args.train_dir, "best.ckpt")
                model.save(save_path)
                best_model = copy.deepcopy(model)
                logging.info("Epoch {:3d}: Save best checkpoint to {}".format(i_epoch, save_path))

        if i_epoch % args.save_every == 0 or i_epoch == args.epochs:
            save_path = os.path.join(args.train_dir, "{}.ckpt".format(i_epoch))
            model.save(save_path)
            logging.info("Epoch {:3d}: Save checkpoint to {}".format(i_epoch, save_path))
    
    ranks = test(test_loader, best_model)
    save_path = os.path.join(args.train_dir, "results_rank.pkl")
    with open(save_path, "wb") as wf:
        pickle.dump(ranks, wf)


if __name__ == "__main__":
    main(sys.argv[1:])
