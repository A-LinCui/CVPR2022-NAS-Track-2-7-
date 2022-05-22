import torch
import torch.nn as nn
from mojito import Mojito
from mojito.mojito import ArchTool
from mojito.tools.constant import *

from aw_nas.evaluator.arch_network import ArchEmbedder
from aw_nas.common import SearchSpace


class Mojito_MBV3_10(SearchSpace, Mojito):
    NAME = "mojito-mbv3-10"

    def __init__(self, schedule_cfg = None):
        super(Mojito_MBV3_10, self).__init__(schedule_cfg)
        Mojito.__init__(self, space = MBV3_10)

    def random_sample(self):
        pass

    def genotype(self, arch):
        pass

    def rollout_from_genotype(self, genotype):
        pass

    def plot_arch(self, genotypes, filename, label, **kwargs):
        pass

    def supported_rollout_types(cls):
        pass

    def distance(self, arch1, arch2):
        pass


class Mojito_LSTMSeqEmbedder(ArchEmbedder):
    r"""
    Examples::
        >>> from aw_nas.common import get_search_space
        >>> search_space = get_search_space("mojito-mbv3-10")
        >>> arch_network = Mojito_LSTMSeqEmbedder(search_space)
        >>> arch = search_space.random_from_design_space()
        >>> arch_embedding = arch_network([arch])[0]
    """
    NAME = "mojito-lstm"

    def __init__(
        self,
        search_space,
        num_hid: int = 100,
        emb_hid: int = 100,
        num_layers: int = 1,
        use_mean: bool = False,
        use_hid: bool = False,
        schedule_cfg = None
    ):
        super(Mojito_LSTMSeqEmbedder, self).__init__(schedule_cfg)
        self.search_space = search_space
        
        self.num_hid = num_hid
        self.num_layers = num_layers
        self.emb_hid = emb_hid
        self.use_mean = use_mean
        self.use_hid = use_hid

        self.choices = {
                "kernel": self.search_space.choices_kernel,
                "expansion": self.search_space.choices_expansion_ratio,
                "depth": self.search_space.choices_depth
        }
        
        self.arch_tool = ArchTool(
            choices_ks = self.choices["kernel"],
            choices_ex = self.choices["expansion"],
            choices_d = self.choices["depth"]
        )
        
        self.num_op = sum([len(choice) for choice in self.choices.values()])
        self.op_emb = nn.Embedding(self.num_op + 1, self.emb_hid)

        self.rnn = nn.LSTM(
            input_size = self.emb_hid,
            hidden_size = self.num_hid,
            num_layers = self.num_layers,
            batch_first = True
        )

        self.out_dim = num_hid

    def trans_to_emb(self, archs: str):
        embs = []
        for arch in archs:
            emb = []
            arch = ArchTool.deserialize(arch) #list
            arch = {
                "kernel": arch[0],
                "expansion": arch[1],
                "depth": arch[2]
            }
            for architecture, choices in zip(arch.values(), self.choices.values()):
                emb += [choices.index(choice) + 1 if choice else 0 for choice in architecture]
            embs.append(emb)
        return embs

    def forward(self, archs):
        x = self.trans_to_emb(archs)
        embs = self.op_emb(torch.LongTensor(x).to(self.op_emb.weight.device))
        out, (h_n, _) = self.rnn(embs)

        if self.use_hid:
            y = h_n[0]
        elif self.use_mean:
            y = torch.mean(out, dim=1)
        else:
            y = out[:, -1, :]
        return y
