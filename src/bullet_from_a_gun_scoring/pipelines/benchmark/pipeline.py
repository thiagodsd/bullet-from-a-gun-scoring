"""
Benchmark pipeline for YOLO v12 experiments
Based on bullet-from-a-gun repository implementation
"""

from kedro.pipeline import node, Pipeline, pipeline
from .nodes import (
    fine_tune_yolo_v12_holes,
    fine_tune_yolo_v12_center,
    evaluate_yolo_v12_holes,
    evaluate_yolo_v12_center,
    compare_experiments
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the benchmark pipeline with YOLO v12 experiments for holes and center detection.
    
    This pipeline includes:
    - Fine-tuning YOLO v12 models for bullet hole detection
    - Fine-tuning YOLO v12 models for center detection
    - Evaluation of both models
    - Comparison of results between experiments
    """
    return pipeline([
        # YOLO v12 Holes Experiment
        node(
            func=fine_tune_yolo_v12_holes,
            inputs="params:yolo_v12_exp1_holes",
            outputs="yolo_v12_holes_training_results",
            name="fine_tune_yolo_v12_holes",
            tags=["yolo", "training", "holes"]
        ),
        node(
            func=evaluate_yolo_v12_holes,
            inputs="params:yolo_v12_exp1_holes",
            outputs=[
                "yolo_v12_holes_evaluation_results",
                "yolo_v12_holes_evaluation_plots"
            ],
            name="evaluate_yolo_v12_holes",
            tags=["yolo", "evaluation", "holes"]
        ),
        
        # YOLO v12 Center Experiment
        node(
            func=fine_tune_yolo_v12_center,
            inputs="params:yolo_v12_exp1_center",
            outputs="yolo_v12_center_training_results",
            name="fine_tune_yolo_v12_center",
            tags=["yolo", "training", "center"]
        ),
        node(
            func=evaluate_yolo_v12_center,
            inputs="params:yolo_v12_exp1_center",
            outputs=[
                "yolo_v12_center_evaluation_results",
                "yolo_v12_center_evaluation_plots"
            ],
            name="evaluate_yolo_v12_center",
            tags=["yolo", "evaluation", "center"]
        ),
        
        # Comparison Analysis
        node(
            func=compare_experiments,
            inputs=[
                "yolo_v12_holes_evaluation_results",
                "yolo_v12_center_evaluation_results"
            ],
            outputs="yolo_v12_comparison_results",
            name="compare_yolo_v12_experiments",
            tags=["comparison", "analysis"]
        )
    ])
