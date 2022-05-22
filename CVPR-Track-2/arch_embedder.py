from typing import List

import torch
import numpy as np
import torch.nn as nn
from extorch.nn.modules import MLP

from aw_nas.evaluator.arch_network import ArchEmbedder


__all__ = [
    "ViT_StraightSeqEmbedder",
    "ViT_SimpleStraightSeqLSTMEmbedder"
]


def simple_convert_arch(archs, search_space, normalize: bool) -> List:
    if normalize:
        max_depth = max(search_space.depth.values())
        min_depth = min(search_space.depth.values())
        max_mlp_ratio = max(search_space.mlp_ratios.values())
        min_mlp_ratio = min(search_space.mlp_ratios.values())
        max_num_heads = max(search_space.num_heads.values())
        min_num_heads = min(search_space.num_heads.values())

    temp_archs = []

    for arch in archs:
        depth = search_space.depth[arch[0]]
        if normalize:
            depth = (depth - min_depth) / (max_depth - min_depth)
        temp_archs.append([depth])  # depth encoding is very important
           
        for i, elm in enumerate(arch):
            if i % 3:
                if (not normalize) or elm == "0":
                    temp_archs[-1].append(int(elm))  # using the original encoding is better than absolute values
                elif i % 3 == 1:
                    num_head = search_space.num_heads[elm]
                    temp_archs[-1].append((num_head - min_num_heads) / (max_num_heads - min_num_heads))
                elif i % 3 == 2:
                    mlp_ratio = search_space.mlp_ratios[elm]
                    temp_archs[-1].append((mlp_ratio - min_mlp_ratio) / (max_mlp_ratio - min_mlp_ratio))

    return temp_archs


class ViT_StraightSeqEmbedder(ArchEmbedder):
    """
    Args:
        normalize (bool): If True, normalize the original values into [0, 1] for architecture encoding.
                          Else, use the encoding index for architecture encoding. Default: ``True``. 
    """
    NAME = "vit-straight"

    def __init__(
        self,
        search_space, 
        normalize: bool = True,
        schedule_cfg = None
    ):
        super(ViT_StraightSeqEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.normalize = normalize
        self.out_dim = 2 * max(self.search_space.depth.values()) + 1
        self._placeholder_tensor = nn.Parameter(torch.zeros(1))
    
    def forward(self, archs):
        converted_archs = simple_convert_arch(archs, self.search_space, self.normalize)
        converted_archs = self._placeholder_tensor.new(converted_archs).unsqueeze(1)
        return converted_archs


class ViT_SimpleStraightSeqLSTMEmbedder(ArchEmbedder):
    """
    Args:
        normalize (bool): If True, normalize the original values into [0, 1] for architecture encoding.
                          Else, use the encoding index for architecture encoding. Default: ``True``.

    Results:
        score: 0.64553
        score_cplfw: 0.25318
        score_market1501: 0.75741
        score_dukemtmc: 0.73419
        score_msmt17: 0.82232
        score_veri: 0.67401
        score_vehicleid: 0.44923
        score_veriwild: 0.70269
        score_sop: 0.77117
    """
    NAME = "vit-simple-straight-lstm"

    def __init__(
        self, 
        search_space, 
        num_hid: int,
        num_layers: int = 1,
        use_mean: bool = False,
        use_hid: bool = False,
        normalize: bool = True,
        schedule_cfg = None
    ):
        super(ViT_SimpleStraightSeqLSTMEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space

        self.emb_dim = 2 * max(self.search_space.depth.values()) + 1

        self.num_hid = num_hid
        self.num_layers = num_layers
        self.use_mean = use_mean
        self.use_hid = use_hid
        self.normalize = normalize
        
        self.rnn = nn.LSTM(
            input_size = self.emb_dim,
            hidden_size = self.num_hid,
            num_layers = self.num_layers,
            batch_first = True
        )

        self.out_dim = num_hid
        self._placeholder_tensor = nn.Parameter(torch.zeros(1))

    def forward(self, archs):
        device = self.rnn._parameters["weight_ih_l0"].device
        converted_archs = simple_convert_arch(archs, self.search_space, self.normalize)
        converted_archs = self._placeholder_tensor.new(converted_archs).unsqueeze(1).to(device)
        out, (h_n, _) = self.rnn(converted_archs)
        if self.use_hid:
            y = h_n[0]
        elif self.use_mean:
            y = torch.mean(out, dim = 1)
        else:
            y = out[:, -1, :]
        return y


class ViT_LearnedTaskEmbeddingStraightSeqLSTMEmbedder(ViT_SimpleStraightSeqLSTMEmbedder):
    """ 
    Results:
        score: 0.64219
        score_cplfw: 0.25593
        score_market1501: 0.75874
        score_dukemtmc: 0.74246
        score_msmt17: 0.82483
        score_veri: 0.63969
        score_vehicleid: 0.45219
        score_veriwild: 0.68945
        score_sop: 0.77424
    """
    NAME = "vit-learned-task-emb-simple-straight-lstm"

    def __init__(
        self,
        search_space, 
        task_num: int,
        num_hid: int,
        num_layers: int = 1,
        use_mean: bool = False,
        use_hid: bool = False,
        normalize: bool = True,
        schedule_cfg = None
    ):
        super(ViT_LearnedTaskEmbeddingStraightSeqLSTMEmbedder, self).__init__(
            search_space, num_hid, num_layers, use_mean, use_hid, normalize, schedule_cfg)
        self.task_embedding = nn.Embedding(task_num, self.emb_dim)

    def forward(self, archs):
        device = self.rnn._parameters["weight_ih_l0"].device
        used_archs = [arch[0] for arch in archs]
        task_ids = [arch[1] for arch in archs]
        converted_archs = simple_convert_arch(used_archs, self.search_space, self.normalize)
        converted_archs = self._placeholder_tensor.new(converted_archs).unsqueeze(1).to(device)
        task_embs = self.task_embedding(torch.tensor(task_ids).to(device)).unsqueeze(1)
        converted_archs = torch.cat((converted_archs, task_embs), 1)
        out, (h_n, _) = self.rnn(converted_archs)
        if self.use_hid:
            y = h_n[0]
        elif self.use_mean:
            y = torch.mean(out, dim = 1)
        else:
            y = out[:, -1, :]
        return y


class ViT_TaskEmbeddingStraightSeqEmbedder(ViT_StraightSeqEmbedder):
    """
    Intuitively, we need task embedding in the arch embedder.
    
    That is because the UFO trains the network with batches of mixture data 
    from different tasks.

    In this embedder, we simply encode the task information with one integer representation.
    """
    NAME = "vit-straight-task-embedding"

    def __init__(
        self,
        search_space, 
        schedule_cfg = None
    ):
        super(ViT_TaskEmbeddingStraightSeqEmbedder, self).__init__(
                search_space, schedule_cfg)

        self.out_dim = 2 * max(self.search_space.depth.values()) + 2

    def _convert_arch(self, archs):
        temp_archs = []
        for arch, task in archs:
            temp_archs.append([task, self.search_space.depth[arch[0]]])
            for i, elm in enumerate(arch):
                if i % 3:
                    temp_archs[-1].append(int(elm))
        return temp_archs


class ViT_LSTMSeqEmbedder(ArchEmbedder):
    NAME = "vit-lstm"

    def __init__(
        self, 
        search_space, 
        head_emb_dim: int,
        mlp_ratio_emb_dim: int,
        num_hid: int,
        num_layers: int = 1,
        use_mean: bool = False,
        use_hid: bool = False,
        schedule_cfg = None
    ):
        super(ViT_LSTMSeqEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space

        self.head_emb_dim = head_emb_dim
        self.mlp_ratio_emb_dim = mlp_ratio_emb_dim
        self.emb_dim = self.head_emb_dim + self.mlp_ratio_emb_dim

        self.num_hid = num_hid
        self.num_layers = num_layers
        self.use_mean = use_mean
        self.use_hid = use_hid
        
        self.head_embedding = nn.Embedding(len(self.search_space.num_heads) + 1, self.head_emb_dim)
        self.mlp_embedding = nn.Embedding(len(self.search_space.mlp_ratios) + 1, self.mlp_ratio_emb_dim)

        self.rnn = nn.LSTM(
            input_size = self.emb_dim,
            hidden_size = self.num_hid,
            num_layers = self.num_layers,
            batch_first = True
        )

        self.out_dim = num_hid

    def forward(self, archs):
        embs = self.embed_and_transform_arch(archs)
        out, (h_n, _) = self.rnn(embs)
        if self.use_hid:
            y = h_n[0]
        elif self.use_mean:
            y = torch.mean(out, dim = 1)
        else:
            y = out[:, -1, :]
        return y

    def embed_and_transform_arch(self, archs):
        all_layer_num = (len(archs[0]) - 1) // 3
        arch_num_heads = []
        arch_mlp_ratios = []

        for arch in archs:
            depth = self.search_space.depth[arch[0]]
            num_heads = [int(arch[3 * i + 1]) for i in range(all_layer_num)]
            arch_num_heads.append(num_heads)
            mlp_ratios = [int(arch[3 * i + 2]) for i in range(all_layer_num)]
            arch_mlp_ratios.append(mlp_ratios)

        arch_num_heads = torch.LongTensor(arch_num_heads).to(self.head_embedding.weight.device)
        arch_mlp_ratios = torch.LongTensor(arch_mlp_ratios).to(self.mlp_embedding.weight.device)

        arch_num_heads_emb = self.head_embedding(arch_num_heads)
        arch_mlp_ratios_emb = self.mlp_embedding(arch_mlp_ratios)

        embs = torch.cat((arch_num_heads_emb, arch_mlp_ratios_emb), 2)
        return embs


class ViT_SimpleSeqLSTMEmbedder(ArchEmbedder):
    NAME = "vit-simple-lstm"

    def __init__(
        self, 
        search_space, 
        num_hid: int,
        num_layers: int = 1,
        use_mean: bool = False,
        use_hid: bool = False,
        schedule_cfg = None
    ):
        super(ViT_SimpleSeqLSTMEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space

        self.emb_dim = max(self.search_space.depth.values())

        self.num_hid = num_hid
        self.num_layers = num_layers
        self.use_mean = use_mean
        self.use_hid = use_hid
        
        self.rnn = nn.LSTM(
            input_size = self.emb_dim,
            hidden_size = self.num_hid,
            num_layers = self.num_layers,
            batch_first = True
        )

        self.out_dim = num_hid
        self._placeholder_tensor = nn.Parameter(torch.zeros(1))

    def forward(self, archs):
        embs = self.embed_and_transform_arch(archs)
        out, (h_n, _) = self.rnn(embs)
        if self.use_hid:
            y = h_n[0]
        elif self.use_mean:
            y = torch.mean(out, dim = 1)
        else:
            y = out[:, -1, :]
        return y

    def embed_and_transform_arch(self, archs):
        all_layer_num = (len(archs[0]) - 1) // 3
        arch_num_heads = []
        arch_mlp_ratios = []

        for arch in archs:
            depth = self.search_space.depth[arch[0]]
            num_heads = [int(arch[3 * i + 1]) for i in range(all_layer_num)]
            arch_num_heads.append(num_heads)
            mlp_ratios = [int(arch[3 * i + 2]) for i in range(all_layer_num)]
            arch_mlp_ratios.append(mlp_ratios)

        device = self.rnn._parameters["weight_ih_l0"].device
        arch_num_heads = self._placeholder_tensor.new(arch_num_heads)
        arch_mlp_ratios = self._placeholder_tensor.new(arch_mlp_ratios)
        embs = torch.cat((arch_num_heads.unsqueeze(1), arch_mlp_ratios.unsqueeze(1)), 1).to(device)
        return embs


class ViT_LSTMSeqSeparateEmbedder(ArchEmbedder):
    """ 
    Currently recommended.

    Results:
        score: 0.71854
        score_cplfw: 0.25758
        score_market1501: 0.83012
        score_dukemtmc: 0.70509
        score_msmt17: 0.79062
        score_veri: 0.85715
        score_vehicleid: 0.66631
        score_veriwild: 0.84704
        score_sop: 0.7944
    """
    NAME = "vit-lstm-separate"

    def __init__(
        self, 
        search_space, 
        head_num_mlp_cfg: dict,
        mlp_ratio_mlp_cfg: dict,
        num_hid: int,
        total_depth_emb_dim: int = None,
        depth_mlp_cfg: dict = None,
        depth_emb_dim: int = None,
        num_layers: int = 1,
        use_mean: bool = False,
        use_hid: bool = False,
        normalize: bool = True, 
        bidirectional: bool = False,
        schedule_cfg = None
    ):
        super(ViT_LSTMSeqSeparateEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space

        self.head_num_mlp_cfg = head_num_mlp_cfg
        self.mlp_ratio_mlp_cfg = mlp_ratio_mlp_cfg
        self.depth_mlp_cfg = depth_mlp_cfg
        self.depth_emb_dim = depth_emb_dim
        self.total_depth_emb_dim = total_depth_emb_dim
        
        self.num_hid = num_hid
        self.num_layers = num_layers
        self.use_mean = use_mean
        self.use_hid = use_hid
        self.normalize = normalize
        self.bidirectional = bidirectional

        self.head_num_mlp = MLP(**self.head_num_mlp_cfg)
        self.mlp_ratio_mlp = MLP(**self.mlp_ratio_mlp_cfg)

        self.emb_dim = self.head_num_mlp.dim_out + self.mlp_ratio_mlp.dim_out

        if self.depth_mlp_cfg:
            self.depth_mlp = MLP(**self.depth_mlp_cfg)
            self.emb_dim += self.depth_mlp.dim_out

        elif self.depth_emb_dim:
            self.depth_embedding = nn.Embedding(
                max(self.search_space.depth.values()), self.depth_emb_dim)
            self.emb_dim += self.depth_emb_dim

        self.rnn = nn.LSTM(
            input_size = self.emb_dim,
            hidden_size = self.num_hid,
            num_layers = self.num_layers,
            batch_first = True,
            bidirectional = self.bidirectional
        )

        self.out_dim = num_hid if not self.bidirectional else 2 * num_hid

        if self.total_depth_emb_dim:
            self.total_depth_embedding = nn.Embedding(
                max(self.search_space.depth.values()), self.total_depth_emb_dim)
            self.out_dim += self.total_depth_emb_dim

    def forward(self, archs):
        embs, arch_total_depth = self.embed_and_transform_arch(archs)
        out, (h_n, _) = self.rnn(embs)
        
        if self.use_hid:
            y = h_n[0]
        elif self.use_mean:
            y = torch.mean(out, dim = 1)
        else:
            y = out[:, -1, :]

        if self.total_depth_emb_dim:
            arch_total_depth_emb = self.total_depth_embedding(arch_total_depth)
            y = torch.cat((y, arch_total_depth_emb), -1)
        return y

    def embed_and_transform_arch(self, archs):
        device = self.head_num_mlp.mlp[0][0].weight.device

        all_layer_num = (len(archs[0]) - 1) // 3
        arch_num_heads = []
        arch_mlp_ratios = []

        arch_total_depth = []

        converted_archs = simple_convert_arch(archs, self.search_space, self.normalize)
        
        for arch in converted_archs:
            arch_total_depth.append(int(2 * arch[0]))
            arch = arch[1:]
            arch_num_heads.append([arch[2 * i] for i in range((len(arch)) // 2)])
            arch_mlp_ratios.append([arch[2 * i + 1] for i in range((len(arch)) // 2)])
        
        arch_num_heads = torch.FloatTensor(arch_num_heads).to(device)
        arch_mlp_ratios = torch.FloatTensor(arch_mlp_ratios).to(device)
        arch_total_depth = torch.LongTensor(arch_total_depth).to(device)

        arch_num_heads_emb = self.head_num_mlp(
            arch_num_heads.unsqueeze(1).view(-1, 1)).view(*arch_num_heads.shape, -1)
        arch_mlp_ratios_emb = self.mlp_ratio_mlp(
            arch_mlp_ratios.unsqueeze(1).view(-1, 1)).view(*arch_mlp_ratios.shape, -1)
        
        embs = torch.cat((arch_num_heads_emb, arch_mlp_ratios_emb), 2)

        if self.depth_mlp_cfg:
            depth_encoding = torch.linspace(0, 1, all_layer_num).repeat(len(embs), 1).to(device)
            depth_encoding_emb = self.depth_mlp(depth_encoding.unsqueeze(1).view(-1, 1)).view(*depth_encoding.shape, -1)
            embs = torch.cat((depth_encoding_emb, embs), -1)
        
        elif self.depth_emb_dim:
            depth_encoding = torch.LongTensor(
                [i for i in range(all_layer_num)]).repeat(len(embs), 1).to(device)
            depth_emb = self.depth_embedding(depth_encoding)
            embs = torch.cat((depth_emb, embs), -1)
        
        return embs, arch_total_depth


class ViT_NewLearnedTaskEmbeddingStraightSeqLSTMEmbedder(ViT_SimpleStraightSeqLSTMEmbedder):
    """ 
    Results:
        score: 0.64219
        score_cplfw: 0.25593
        score_market1501: 0.75874
        score_dukemtmc: 0.74246
        score_msmt17: 0.82483
        score_veri: 0.63969
        score_vehicleid: 0.45219
        score_veriwild: 0.68945
        score_sop: 0.77424
    """
    NAME = "vit-learned-task-emb-simple-straight-lstm-new"

    def __init__(
        self,
        search_space, 
        task_num: int,
        num_hid: int,
        num_layers: int = 1,
        task_emb_dim: int = 100,
        use_mean: bool = False,
        use_hid: bool = False,
        normalize: bool = True,
        schedule_cfg = None
    ):
        super(ViT_NewLearnedTaskEmbeddingStraightSeqLSTMEmbedder, self).__init__(
            search_space, num_hid, num_layers, use_mean, use_hid, normalize, schedule_cfg)

        self.task_embedding = nn.Embedding(task_num, task_emb_dim)
        self.out_dim = num_hid + task_emb_dim

    def forward(self, archs):
        device = self.rnn._parameters["weight_ih_l0"].device
        used_archs = [arch[0] for arch in archs]
        task_ids = [arch[1]  for arch in archs]
        converted_archs = simple_convert_arch(used_archs, self.search_space, self.normalize)
        converted_archs = self._placeholder_tensor.new(converted_archs).unsqueeze(1).to(device)
        out, (h_n, _) = self.rnn(converted_archs)
             
        if self.use_hid:
            y = h_n[0]
        elif self.use_mean:
            y = torch.mean(out, dim = 1)
        else:
            y = out[:, -1, :]
        
        task_embs = self.task_embedding(torch.tensor(task_ids).to(device)).unsqueeze(1)
        y = torch.cat((y, task_embs.squeeze(1)), 1)
        return y



class ViT_LSTMSeqSeparateEmbeddingEmbedder(ArchEmbedder):
    """ 

    Not recommended.

    Results:
        score: 
        score_cplfw:
        score_market1501: 
        score_dukemtmc: 
        score_msmt17: 
        score_veri: 
        score_vehicleid:
        score_veriwild: 
        score_sop: 
    """
    NAME = "vit-lstm-separate-embedding"

    def __init__(
        self, 
        search_space, 
        head_num_emb_dim: int,
        mlp_ratio_emb_dim: int,
        num_hid: int,
        embedding_simple_concat_depth: bool = False,
        num_layers: int = 1,
        use_mean: bool = False,
        use_hid: bool = False,
        normalize: bool = True, 
        schedule_cfg = None
    ):
        super(ViT_LSTMSeqSeparateEmbeddingEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space

        self.num_hid = num_hid
        self.num_layers = num_layers
        self.use_mean = use_mean
        self.use_hid = use_hid
        self.normalize = normalize
        self.embedding_simple_concat_depth = embedding_simple_concat_depth

        self.head_num_emb = nn.Embedding(
            len(self.search_space.num_heads.keys()) + 1, head_num_emb_dim)
        self.mlp_ratio_emb = nn.Embedding(
            len(self.search_space.mlp_ratios.keys()) + 1, mlp_ratio_emb_dim)

        self.emb_dim = head_num_emb_dim + mlp_ratio_emb_dim \
                + bool(self.embedding_simple_concat_depth)
        
        self.rnn = nn.LSTM(
            input_size = self.emb_dim,
            hidden_size = self.num_hid,
            num_layers = self.num_layers,
            batch_first = True
        )

        self.out_dim = num_hid

    def forward(self, archs):
        embs = self.embed_and_transform_arch(archs)
        out, (h_n, _) = self.rnn(embs)
        if self.use_hid:
            y = h_n[0]
        elif self.use_mean:
            y = torch.mean(out, dim = 1)
        else:
            y = out[:, -1, :]
        return y

    def embed_and_transform_arch(self, archs):
        device = self.rnn._parameters["weight_ih_l0"].device

        all_layer_num = (len(archs[0]) - 1) // 3
        arch_num_heads = []
        arch_mlp_ratios = []

        converted_archs = simple_convert_arch(archs, self.search_space, False)
        
        for arch in converted_archs:
            arch = arch[1:]
            arch_num_heads.append([arch[2 * i] for i in range((len(arch)) // 2)])
            arch_mlp_ratios.append([arch[2 * i + 1] for i in range((len(arch)) // 2)])
        
        arch_num_heads = torch.LongTensor(arch_num_heads).to(device)
        arch_mlp_ratios = torch.LongTensor(arch_mlp_ratios).to(device)

        arch_num_heads_emb = self.head_num_emb(arch_num_heads)
        arch_mlp_ratios_emb = self.mlp_ratio_emb(arch_mlp_ratios)

        embs = torch.cat((arch_num_heads_emb, arch_mlp_ratios_emb), 2)

        import ipdb
        ipdb.set_trace()
        return embs


class ViT_LearnedTaskEmbeddingLSTMSeqSeparateEmbedder(ViT_LSTMSeqSeparateEmbedder):
    NAME = "vit-lstm-separate-learned_task_emb"

    def __init__(
        self, 
        search_space, 
        head_num_mlp_cfg: dict,
        mlp_ratio_mlp_cfg: dict,
        num_hid: int,
        task_num: int,
        task_emb_dim: int,
        depth_mlp_cfg: dict = None,
        depth_emb_dim: int = None,
        num_layers: int = 1,
        use_mean: bool = False,
        use_hid: bool = False,
        normalize: bool = True, 
        schedule_cfg = None
    ):
        super(ViT_LearnedTaskEmbeddingLSTMSeqSeparateEmbedder, self).__init__(
                search_space, head_num_mlp_cfg, mlp_ratio_mlp_cfg, num_hid, 
                depth_mlp_cfg, depth_emb_dim, num_layers, use_mean, use_hid, normalize, schedule_cfg)

        self.task_embedding = nn.Embedding(task_num, task_emb_dim)
        self.out_dim += task_emb_dim

    def forward(self, archs):
        used_archs = [arch[0] for arch in archs]
        task_ids = [arch[1] for arch in archs]
        
        embs = self.embed_and_transform_arch(used_archs)

        out, (h_n, _) = self.rnn(embs)
        if self.use_hid:
            y = h_n[0]
        elif self.use_mean:
            y = torch.mean(out, dim = 1)
        else:
            y = out[:, -1, :]

        task_embs = self.task_embedding(torch.tensor(task_ids).to(self.head_num_mlp.mlp[0][0].weight.device)).unsqueeze(1)
        y = torch.cat((y, task_embs.squeeze(1)), 1)
        return y

    def embed_and_transform_arch(self, archs):
        device = self.head_num_mlp.mlp[0][0].weight.device

        all_layer_num = (len(archs[0]) - 1) // 3
        arch_num_heads = []
        arch_mlp_ratios = []

        converted_archs = simple_convert_arch(archs, self.search_space, self.normalize)
        
        for arch in converted_archs:
            arch = arch[1:]
            arch_num_heads.append([arch[2 * i] for i in range((len(arch)) // 2)])
            arch_mlp_ratios.append([arch[2 * i + 1] for i in range((len(arch)) // 2)])
        
        arch_num_heads = torch.FloatTensor(arch_num_heads).to(device)
        arch_mlp_ratios = torch.FloatTensor(arch_mlp_ratios).to(device)

        arch_num_heads_emb = self.head_num_mlp(
            arch_num_heads.unsqueeze(1).view(-1, 1)).view(*arch_num_heads.shape, -1)
        arch_mlp_ratios_emb = self.mlp_ratio_mlp(
            arch_mlp_ratios.unsqueeze(1).view(-1, 1)).view(*arch_mlp_ratios.shape, -1)
        
        embs = torch.cat((arch_num_heads_emb, arch_mlp_ratios_emb), 2)

        if self.depth_mlp_cfg:
            depth_encoding = torch.linspace(0, 1, all_layer_num).repeat(len(embs), 1).to(device)
            depth_encoding_emb = self.depth_mlp(depth_encoding.unsqueeze(1).view(-1, 1)).view(*depth_encoding.shape, -1)
            embs = torch.cat((depth_encoding_emb, embs), -1)
        
        elif self.depth_emb_dim:
            depth_encoding = torch.LongTensor(
                [i for i in range(all_layer_num)]).repeat(len(embs), 1).to(device)
            depth_emb = self.depth_embedding(depth_encoding)
            embs = torch.cat((depth_emb, embs), -1)
        
        return embs


class ViT_LearnedTaskEmbeddingBeforeLSTMSeqSeparateEmbedder(ViT_LSTMSeqSeparateEmbedder):
    NAME = "vit-lstm-separate-learned_task_emb_before"

    def __init__(
        self, 
        search_space, 
        head_num_mlp_cfg: dict,
        mlp_ratio_mlp_cfg: dict,
        num_hid: int,
        task_num: int,
        task_emb_dim: int,
        depth_mlp_cfg: dict = None,
        depth_emb_dim: int = None,
        num_layers: int = 1,
        use_mean: bool = False,
        use_hid: bool = False,
        normalize: bool = True, 
        schedule_cfg = None
    ):
        super(ViT_LearnedTaskEmbeddingBeforeLSTMSeqSeparateEmbedder, self).__init__(
                search_space, head_num_mlp_cfg, mlp_ratio_mlp_cfg, num_hid, 
                depth_mlp_cfg, depth_emb_dim, num_layers, use_mean, use_hid, normalize, schedule_cfg)
        
        self.task_embedding = nn.Embedding(task_num, task_emb_dim)
        self.emb_dim += task_emb_dim

        self.rnn = nn.LSTM(
            input_size = self.emb_dim,
            hidden_size = self.num_hid,
            num_layers = self.num_layers,
            batch_first = True
        )

    def forward(self, archs):
        used_archs = [arch[0] for arch in archs]
        task_ids = [arch[1] for arch in archs]
        
        embs = self.embed_and_transform_arch(used_archs)
        task_embs = self.task_embedding(
            torch.tensor(task_ids).to(self.head_num_mlp.mlp[0][0].weight.device)).unsqueeze(1)
        task_embs = task_embs.repeat(1, embs.shape[1], 1)
        embs = torch.cat((embs, task_embs), -1)

        out, (h_n, _) = self.rnn(embs)
        if self.use_hid:
            y = h_n[0]
        elif self.use_mean:
            y = torch.mean(out, dim = 1)
        else:
            y = out[:, -1, :]

        return y

    def embed_and_transform_arch(self, archs):
        device = self.head_num_mlp.mlp[0][0].weight.device

        all_layer_num = (len(archs[0]) - 1) // 3
        arch_num_heads = []
        arch_mlp_ratios = []

        converted_archs = simple_convert_arch(archs, self.search_space, self.normalize)
        
        for arch in converted_archs:
            arch = arch[1:]
            arch_num_heads.append([arch[2 * i] for i in range((len(arch)) // 2)])
            arch_mlp_ratios.append([arch[2 * i + 1] for i in range((len(arch)) // 2)])
        
        arch_num_heads = torch.FloatTensor(arch_num_heads).to(device)
        arch_mlp_ratios = torch.FloatTensor(arch_mlp_ratios).to(device)

        arch_num_heads_emb = self.head_num_mlp(
            arch_num_heads.unsqueeze(1).view(-1, 1)).view(*arch_num_heads.shape, -1)
        arch_mlp_ratios_emb = self.mlp_ratio_mlp(
            arch_mlp_ratios.unsqueeze(1).view(-1, 1)).view(*arch_mlp_ratios.shape, -1)
        
        embs = torch.cat((arch_num_heads_emb, arch_mlp_ratios_emb), 2)

        if self.depth_mlp_cfg:
            depth_encoding = torch.linspace(0, 1, all_layer_num).repeat(len(embs), 1).to(device)
            depth_encoding_emb = self.depth_mlp(depth_encoding.unsqueeze(1).view(-1, 1)).view(*depth_encoding.shape, -1)
            embs = torch.cat((depth_encoding_emb, embs), -1)
        
        elif self.depth_emb_dim:
            depth_encoding = torch.LongTensor(
                [i for i in range(all_layer_num)]).repeat(len(embs), 1).to(device)
            depth_emb = self.depth_embedding(depth_encoding)
            embs = torch.cat((depth_emb, embs), -1)
        
        return embs



class ViT_LSTMSeqSeparateTestEmbedder(ArchEmbedder):
    NAME = "vit-lstm-separate-test"

    def __init__(
        self, 
        search_space, 
        head_num_mlp_cfg: dict,
        mlp_ratio_mlp_cfg: dict,
        num_hid: int,
        depth_emb_dim: int = None,
        num_layers: int = 1,
        use_mean: bool = False,
        use_hid: bool = False,
        normalize: bool = True, 
        schedule_cfg = None
    ):
        super(ViT_LSTMSeqSeparateTestEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space

        self.head_num_mlp_cfg = head_num_mlp_cfg
        self.mlp_ratio_mlp_cfg = mlp_ratio_mlp_cfg
        self.depth_emb_dim = depth_emb_dim
        
        self.num_hid = num_hid
        self.num_layers = num_layers
        self.use_mean = use_mean
        self.use_hid = use_hid
        self.normalize = normalize

        self.head_num_mlp = MLP(dim_in = 1 + self.depth_emb_dim, **self.head_num_mlp_cfg)
        self.mlp_ratio_mlp = MLP(dim_in = 1 + self.depth_emb_dim, **self.mlp_ratio_mlp_cfg)

        self.emb_dim = self.head_num_mlp.dim_out + self.mlp_ratio_mlp.dim_out

        self.arch_num_depth_embedding = nn.Embedding(
            max(self.search_space.depth.values()), self.depth_emb_dim)
        self.mlp_ratio_depth_embedding = nn.Embedding(
            max(self.search_space.depth.values()), self.depth_emb_dim)
        
        
        self.rnn = nn.LSTM(
            input_size = self.emb_dim,
            hidden_size = self.num_hid,
            num_layers = self.num_layers,
            batch_first = True
        )

        self.out_dim = num_hid

    def forward(self, archs):
        embs = self.embed_and_transform_arch(archs)
        out, (h_n, _) = self.rnn(embs)
        if self.use_hid:
            y = h_n[0]
        elif self.use_mean:
            y = torch.mean(out, dim = 1)
        else:
            y = out[:, -1, :]
        return y

    def embed_and_transform_arch(self, archs):
        device = self.head_num_mlp.mlp[0][0].weight.device

        all_layer_num = (len(archs[0]) - 1) // 3
        arch_num_heads = []
        arch_mlp_ratios = []

        converted_archs = simple_convert_arch(archs, self.search_space, self.normalize)
        
        for arch in converted_archs:
            arch = arch[1:]
            arch_num_heads.append([arch[2 * i] for i in range((len(arch)) // 2)])
            arch_mlp_ratios.append([arch[2 * i + 1] for i in range((len(arch)) // 2)])
        
        arch_num_heads = torch.FloatTensor(arch_num_heads).to(device)
        arch_mlp_ratios = torch.FloatTensor(arch_mlp_ratios).to(device)

        depth_encoding = torch.LongTensor(
            [i for i in range(all_layer_num)]).repeat(len(arch_num_heads), 1).to(device)
        arch_num_depth_emb = self.arch_num_depth_embedding(depth_encoding)
        mlp_ratio_depth_emb = self.mlp_ratio_depth_embedding(depth_encoding)

        arch_num_heads_emb = self.head_num_mlp(torch.cat((arch_num_heads.unsqueeze(-1), arch_num_depth_emb), -1))
        arch_mlp_ratios_emb = self.mlp_ratio_mlp(torch.cat((arch_mlp_ratios.unsqueeze(-1), mlp_ratio_depth_emb), -1))

        embs = torch.cat((arch_num_heads_emb, arch_mlp_ratios_emb), 2)

        return embs
