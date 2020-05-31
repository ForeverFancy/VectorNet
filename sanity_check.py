import unittest
from model import SubGraph
from model import GlobalGraph
from data_process import *
import torch


class SubGraphTest(unittest.TestCase):
    def check_forward(self):
        model = SubGraph()
        features, mask = load_features()
        model.train()
        print(len(features))

        features = torch.from_numpy(features).to(dtype=torch.float)
        mask = torch.from_numpy(mask).to(dtype=torch.float)
        print(mask.shape)
        print(features.shape)
        # print(mask)
        # print(mask.unsqueeze(-1).repeat(1,1,2))
        out = model.forward(features, mask)
        print(out.shape)


class GlobalGraphTest(unittest.TestCase):
    def check_forward(self):
        submodel = SubGraph()
        globalmodel = GlobalGraph()
        features, mask = load_features()
        submodel.train()
        globalmodel.train()
        features = torch.from_numpy(features).to(dtype=torch.float)
        mask = torch.from_numpy(mask).to(dtype=torch.float)
        print(features.shape)
        out = submodel.forward(features, mask)
        # print(out.shape)
        print(out.shape)
        out = globalmodel.forward(out)
        print(out.shape)


if __name__ == "__main__":
    unittest.TextTestRunner().run(GlobalGraphTest("check_forward"))
