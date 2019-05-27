from torch import nn
from torch.nn import functional as F
from ..box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from maskrcnn_benchmark.modeling.poolers import Pooler


class FPN2MLP2FeatureExtractor(nn.Module):
    """
    Heads for FPN for car classification and rotation estimation
    """
    def __init__(self, cfg):
        super().__init__()
        resolution = cfg.MODEL.ROI_CAR_CLS_ROT_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_CAR_CLS_ROT_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_CAR_CLS_ROT_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS * resolution ** 2
        representation_size = cfg.MODEL.ROI_CAR_CLS_ROT_HEAD.MLP_HEAD_DIM
        self.pooler = pooler
        self.fc6 = nn.Linear(input_size, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

        for l in [self.fc6, self.fc7]:
            # Caffe2 implementation uses XavierFill, which in fact
            # corresponds to kaiming_uniform_ in PyTorch
            nn.init.kaiming_uniform_(l.weight)
            nn.init.constant_(l.bias, 0)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x), inplace=True)
        x = F.relu(self.fc7(x), inplace=True)

        return x


_ROI_CAR_CLS_ROT_FEATURE_EXTRACTORS = {
    "ResNet50Conv5ROIFeatureExtractor": ResNet50Conv5ROIFeatureExtractor,
    "FPN2MLP2FeatureExtractor": FPN2MLP2FeatureExtractor,
}


def make_roi_car_cls_rot_feature_extractor(cfg):
    func = _ROI_CAR_CLS_ROT_FEATURE_EXTRACTORS[cfg.MODEL.ROI_CAR_CLS_ROT_HEAD.FEATURE_EXTRACTOR]
    return func(cfg)
