from typing import List, Dict

from aw_nas.common import SearchSpace


class ViTSearchSpace(SearchSpace):
    NAME = "vit"

    def __init__(self, 
        depth: Dict[str, int] = {"j": 10, "k": 11, "l": 12}, 
        num_heads: Dict[str, int] = {"1": 12, "2": 11, "3": 10}, 
        mlp_ratios: Dict[str, float] = {"1": 4.0, "2": 3.5, "3": 3.0},
        schedule_cfg = None
    ):
        super(ViTSearchSpace, self).__init__(schedule_cfg)
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratios = mlp_ratios

    def convert_arch(self, arch):
        depth = self.depth[arch[0]]
        num_heads = [self.num_heads[arch[i + 1]] for i in range(depth)]
        mlp_ratios = [self.mlp_ratios[arch[i + 2]] for i in range(depth)]
        return depth, num_heads, mlp_ratios
    
    def random_sample(self):
        pass

    def genotype(self, arch):
        pass

    def rollout_from_genotype(self, genotype):
        pass

    def plot_arch(self, genotypes, filename, label, **kwargs):
        pass

    def distance(self, arch1, arch2):
        pass

    def supported_rollout_types(cls):
        pass
