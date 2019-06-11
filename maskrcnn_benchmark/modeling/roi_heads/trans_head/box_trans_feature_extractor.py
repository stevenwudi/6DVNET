from torch import nn
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.roi_heads.nn_init import XavierFill, MSRAFill


class MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for car classification and rotation estimation
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.dim_in = cfg.MODEL.TRANS_HEAD.INPUT_DIM
        self.hidden_dim = cfg.MODEL.TRANS_HEAD.MLP_HEAD_DIM

        self.fc1 = nn.Linear(self.dim_in, self.hidden_dim)
        self.fc1_bn = nn.BatchNorm1d(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2_bn = nn.BatchNorm1d(self.hidden_dim)

        nn.init.constant_(self.fc1_bn.weight, 1.0)
        nn.init.constant_(self.fc2_bn.weight, 1.0)

        for l in [self.fc1, self.fc1]:
            # Caffe2 implementation uses XavierFill, which in fact
            # corresponds to kaiming_uniform_ in PyTorch
            nn.init.kaiming_uniform_(l.weight, a=1)
            #XavierFill(l.weight)
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.fc1_bn(self.fc1(x.view(batch_size, -1))), inplace=True)
        x = F.relu(self.fc2_bn(self.fc2(x)), inplace=True)
        return x


_ROI_BOX_TRANS_FEATURE_EXTRACTORS = {"MLPFeatureExtractor": MLPFeatureExtractor, }


def make_box_trans_feature_extractor(cfg):
    func = _ROI_BOX_TRANS_FEATURE_EXTRACTORS[cfg.MODEL.TRANS_HEAD.FEATURE_EXTRACTOR]
    return func(cfg)



