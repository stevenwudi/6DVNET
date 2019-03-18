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
        output_folder=output_folder,
        cfg=cfg,
    )
