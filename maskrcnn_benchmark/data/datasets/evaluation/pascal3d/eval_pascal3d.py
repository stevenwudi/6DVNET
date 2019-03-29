import os
from maskrcnn_benchmark.data.datasets.evaluation.apollo_3d_car_instace.wad_eval import WAD_eval
import pickle


def do_pascal3d_evaluation(
    dataset,
    predictions,
    output_folder,
    cfg,
    vis=False
):

    # We First evaluate the box, segm and then evaluate the 3D metric
    # First boxes
    eval_type = 'boxes'
    # Since reading GT need to retrieve from the hard disc, so we try to cache it.
    coco_gt = dataset.loadGt(type=eval_type)
    coco_dt = dataset.loadRes(predictions, type=eval_type)

    coco_eval = WAD_eval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    """
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=all                  | maxDets=100 ] = 0.870
     Average Precision  (AP) @[ IoU=0.50      | area=all                  | maxDets=100 ] = 0.988
     Average Precision  (AP) @[ IoU=0.75      | area=all                  | maxDets=100 ] = 0.964
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=extra-small (0-14)   | maxDets=100 ] = -1.000 | (numGT, numDt) =     0    23
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium(28-56)        | maxDets=100 ] = 0.098 | (numGT, numDt) =    10   247
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=large(56-112)        | maxDets=100 ] = 0.380 | (numGT, numDt) =    45   472
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=extra-large(112-512) | maxDets=100 ] = 0.883 | (numGT, numDt) =  2622  3125
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=uber-large(512 !!!!) | maxDets=100 ] = 0.712 | (numGT, numDt) =   103   249
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=all                  | maxDets=  1 ] = 0.888
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=all                  | maxDets= 10 ] = 0.901
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=all                  | maxDets=100 ] = 0.901
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=extra-small (0-14)   | maxDets=100 ] = -1.000 | (numGT, numDt) =     0    23
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium(28-56)        | maxDets=100 ] = 0.290 | (numGT, numDt) =    10   247
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=large(56-112)        | maxDets=100 ] = 0.578 | (numGT, numDt) =    45   472
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=extra-large(112-512) | maxDets=100 ] = 0.914 | (numGT, numDt) =  2622  3125
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=uber-large(512 !!!!) | maxDets=100 ] = 0.754 | (numGT, numDt) =   103   249
    """

    # Now we evaluate the segm
    eval_type = 'segms'
    gt_file = os.path.join(output_folder, 'val_gt_' + eval_type + '.pth')
    # Since reading GT need to retrieve from the hard disc, so we try to cache it.
    if os.path.isfile(gt_file):
        coco_gt = pickle.load(open(gt_file, "rb"))
    else:
        coco_gt = dataset.loadGt(type=eval_type)
        pickle.dump(coco_gt, open(gt_file, "wb"))

    dt_file = os.path.join(output_folder, 'val_dt_' + eval_type + '.pth')
    if os.path.isfile(gt_file):
        coco_dt = pickle.load(open(dt_file, "rb"))
    else:
        coco_dt = dataset.loadRes(predictions, type=eval_type)
        pickle.dump(coco_dt, open(dt_file, "wb"))

    coco_eval = WAD_eval(coco_gt, coco_dt, 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()

