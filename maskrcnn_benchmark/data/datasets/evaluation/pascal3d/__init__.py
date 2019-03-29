from .eval_pascal3d import do_pascal3d_evaluation


def pascal3d_evaluation(
    dataset,
    predictions,
    output_folder,
    cfg,
    box_only,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    return do_pascal3d_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        cfg=cfg,
    )
