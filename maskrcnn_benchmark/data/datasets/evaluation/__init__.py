from maskrcnn_benchmark.data import datasets

from .coco import coco_evaluation
from .voc import voc_evaluation
from .kitti import kitti_evaluation
from .apollo_3d_car_instace import car_3d_evaluation
from .pascal3d import pascal3d_evaluation


def evaluate(dataset, predictions, output_folder, cfg, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(dataset=dataset, predictions=predictions, output_folder=output_folder, cfg=cfg, **kwargs)
    if isinstance(dataset, datasets.COCODataset):
        return coco_evaluation(**args)
    elif isinstance(dataset, datasets.PascalVOCDataset):
        return voc_evaluation(**args)
    elif isinstance(dataset, datasets.KittiInstanceDataset):
        return kitti_evaluation(**args)
    elif isinstance(dataset, datasets.Car3D):
        return car_3d_evaluation(**args)
    elif isinstance(dataset, datasets.Pascal3D):
        return pascal3d_evaluation(**args)

    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
