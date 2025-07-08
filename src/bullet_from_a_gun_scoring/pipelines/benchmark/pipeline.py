"""
Benchmark pipeline using modular pipelines approach
Based on bullet-from-a-gun repository implementation
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    fine_tune_yolo,
    evaluate_yolo,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the benchmark pipeline using modular pipelines.
    
    This creates template pipelines that can be reused across different experiments
    with proper namespacing for parameter isolation.
    
    Returns:
        Pipeline: The complete benchmark pipeline with all experiment variants
    """
    
    # Template YOLO pipeline
    template_yolo = pipeline([
        node(
            func=fine_tune_yolo,
            inputs=[
                "params:dataprep_params",
                "params:fine_tuning_params",
            ],
            outputs="fine_tuning_results",
            name="fine_tune_yolo",
        ),
        node(
            func=evaluate_yolo,
            inputs=[
                "params:dataprep_params",
                "params:fine_tuning_params",
            ],
            outputs=[
                "evaluation_results",
                "evaluation_plots",
            ],
            name="evaluate_yolo",
        ),
    ])
    
    # Create specific experiment pipelines using namespaces
    yolo_v12_exp1_holes = pipeline(
        pipe=template_yolo,
        namespace="yolo.yolo_v12_exp1_holes",
    )
    # kedro run --pipeline benchmark -n yolo.yolo_v12_exp1_holes.fine_tune_yolo
    # kedro run --pipeline benchmark -n yolo.yolo_v12_exp1_holes.evaluate_yolo
    
    yolo_v12_exp1_center = pipeline(
        pipe=template_yolo,
        namespace="yolo.yolo_v12_exp1_center",
    )
    # kedro run --pipeline benchmark -n yolo.yolo_v12_exp1_center.fine_tune_yolo
    # kedro run --pipeline benchmark -n yolo.yolo_v12_exp1_center.evaluate_yolo
    
    return yolo_v12_exp1_holes + yolo_v12_exp1_center