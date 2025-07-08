"""
Benchmark pipeline nodes for YOLO v12 experiments
Based on bullet-from-a-gun repository implementation
"""

import json
import logging
import os
import random
from pathlib import Path  # noqa: F401
from typing import Any, Dict, Tuple  # noqa: UP035

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def setup_cuda_environment():
    """Setup CUDA environment for training"""
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        logger.info(f"CUDA Device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    else:
        logger.warning("CUDA not available, using CPU")


def setup_reproducibility(seed: int = 0):
    """Setup reproducible random seeds"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.set_printoptions(precision=5)


def fine_tune_yolo(
    dataprep_params: Dict[str, Any],
    fine_tuning_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Fine-tune YOLO model for object detection.
    
    Args:
        dataprep_params: Data preparation parameters
        fine_tuning_params: Fine-tuning parameters
        
    Returns:
        Dictionary with training results
    """
    
    # Setup environment
    setup_reproducibility()
    setup_cuda_environment()
    
    # Setup paths
    experiment_id = dataprep_params['experiment_id']
    output_path = os.path.join(*fine_tuning_params["path"])
    yolo_data_path = os.path.join(*dataprep_params['yolo_data']['path'])
    yolo_config_path = os.path.join(yolo_data_path, 'data.yaml')
    
    logger.info(f"Training YOLO v12 for experiment: {experiment_id}")
    logger.info(f"Data path: {yolo_data_path}")
    logger.info(f"Output path: {output_path}")
    
    # Load YOLO model
    model = YOLO(
        fine_tuning_params["model_name"].replace(".pt", ".yaml")
    ).load(
        fine_tuning_params["model_name"]
    )
    
    if torch.cuda.is_available():
        model.to("cuda")
    
    # Train model
    results = model.train(
        data=yolo_config_path,
        epochs=fine_tuning_params["model_config"]["epochs"],
        batch=fine_tuning_params["model_config"]["batch"],
        imgsz=fine_tuning_params["model_config"]["img_size"],
        project=output_path,
        name=experiment_id,
        optimizer=fine_tuning_params["model_config"]["optimizer"],
        lr0=fine_tuning_params["model_config"]["lr0"],
        device=0 if torch.cuda.is_available() else "cpu",
        exist_ok=True,
        save=True,
        val=True,
        cache=True,
        amp=False,
        plots=True,
    )
    
    return {"training_completed": True, "experiment_id": experiment_id}




def evaluate_yolo(
    dataprep_params: Dict[str, Any],
    fine_tuning_params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Evaluate YOLO model performance.
    
    Args:
        dataprep_params: Data preparation parameters
        fine_tuning_params: Fine-tuning parameters
        
    Returns:
        Tuple of (evaluation results, plot paths)
    """
    
    # Setup environment
    setup_reproducibility()
    setup_cuda_environment()
    
    # Setup paths
    experiment_id = dataprep_params['experiment_id']
    output_path = os.path.join(*fine_tuning_params["path"])
    yolo_data_path = os.path.join(*dataprep_params['yolo_data']['path'])
    yolo_config_path = os.path.join(yolo_data_path, 'data.yaml')
    
    # Load trained model
    model_path = os.path.join(output_path, experiment_id, "weights", "best.pt")
    model = YOLO(model_path)
    
    if torch.cuda.is_available():
        model.to("cuda")
    
    logger.info(f"Evaluating YOLO v12 for experiment: {experiment_id}")
    
    results = {}
    plots = {}
    
    # Evaluate on each dataset split
    for split in ["train", "val", "test"]:
        logger.info(f"Evaluating on {split} split")
        
        # Run validation
        metrics = model.val(
            data=yolo_config_path,
            batch=fine_tuning_params["model_config"]["batch"],
            imgsz=fine_tuning_params["model_config"]["img_size"],
            iou=fine_tuning_params["model_config"]["iou"],
            project=output_path,
            name=f"{experiment_id}_{split}",
            save_json=True,
            exist_ok=True,
            plots=True,
            split=split,
        )
        
        # Process metrics
        metrics_dict = {
            "results": metrics.results_dict,
            "confusion_matrix": {
                "tp": metrics.confusion_matrix.tp_fp()[0].tolist(),
                "fp": metrics.confusion_matrix.tp_fp()[1].tolist(),
                "matrix": metrics.confusion_matrix.matrix.tolist()
            } if metrics.confusion_matrix is not None else None
        }
        
        results[split] = metrics_dict
        
        # Save detailed results
        output_folder = os.path.join(output_path, f"{experiment_id}_{split}")
        os.makedirs(output_folder, exist_ok=True)
        
        with open(os.path.join(output_folder, "metrics.json"), "w") as f:
            json.dump(metrics_dict, f, indent=4)
        
        # Generate predictions for visualization
        images_folder = os.path.join(yolo_data_path, split, "images")
        if os.path.exists(images_folder):
            _generate_predictions(model, images_folder, output_folder, fine_tuning_params)
    
    return results, plots




def _generate_predictions(model, images_folder: str, output_folder: str, fine_tuning_params: Dict):
    """Generate predictions for visualization"""
    logger.info(f"Generating predictions for images in {images_folder}")
    
    # Get image files
    image_files = [
        os.path.join(images_folder, img) 
        for img in os.listdir(images_folder) 
        if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]
    
    # Create predictions folder
    predictions_folder = os.path.join(output_folder, "predictions")
    os.makedirs(predictions_folder, exist_ok=True)
    
    # Process each image
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        
        # Run prediction
        result = model(img_path, imgsz=fine_tuning_params["model_config"]["img_size"])[0]
        
        # Save visualization
        result.save(filename=os.path.join(predictions_folder, f"pred_{img_name}"))
        
        # Save predictions to JSON
        boxes_data = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                box_info = {
                    "class": int(box.cls.cpu().numpy()[0]),
                    "confidence": float(box.conf.cpu().numpy()[0]),
                    "bbox": box.xyxy.cpu().numpy()[0].tolist()
                }
                boxes_data.append(box_info)
        
        # Save JSON
        json_filename = os.path.splitext(img_name)[0] + '.json'
        with open(os.path.join(predictions_folder, json_filename), 'w') as f:
            json.dump(boxes_data, f, indent=4)
    
    logger.info(f"Generated predictions for {len(image_files)} images")


