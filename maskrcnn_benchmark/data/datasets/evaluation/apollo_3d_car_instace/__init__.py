from .eval_car_instances import do_3d_car_instance_evaluation


def car_3d_evaluation(
    dataset,
    predictions,
    output_folder,
    cfg,
    box_only,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    return do_3d_car_instance_evaluation(
        dataset=dataset,
        predictions=predictions,
        box_only=box_only,
        output_folder=output_folder,
        cfg=cfg,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )
