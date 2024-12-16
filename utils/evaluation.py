"""
Evaluation utilities for solar panel detection.
"""

import numpy as np
from typing import List, Dict, Any, Tuple

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes.
    
    Args:
        box1 (np.ndarray): First box coordinates [x1, y1, x2, y2]
        box2 (np.ndarray): Second box coordinates [x1, y1, x2, y2]
        
    Returns:
        float: IoU score
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def calculate_map(predictions: List[Dict[str, Any]], 
                 ground_truth: List[Dict[str, Any]], 
                 iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate mean Average Precision (mAP) for object detection.
    
    Args:
        predictions (list): List of prediction dictionaries
        ground_truth (list): List of ground truth dictionaries
        iou_threshold (float): IoU threshold for considering a detection as correct
        
    Returns:
        dict: Dictionary containing mAP and other metrics
    """
    # Initialize metrics
    metrics = {
        'mAP': 0.0,
        'precision': 0.0,
        'recall': 0.0
    }
    
    true_positives = 0
    false_positives = 0
    total_gt = sum(len(gt['boxes']) for gt in ground_truth)
    
    # Match predictions to ground truth
    for pred, gt in zip(predictions, ground_truth):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        gt_boxes = gt['boxes']
        
        # Sort predictions by confidence
        sorted_idx = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sorted_idx]
        
        # Match each prediction to ground truth
        for pred_box in pred_boxes:
            matched = False
            for gt_box in gt_boxes:
                iou = calculate_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    matched = True
                    true_positives += 1
                    break
            
            if not matched:
                false_positives += 1
    
    # Calculate metrics
    if true_positives + false_positives > 0:
        metrics['precision'] = true_positives / (true_positives + false_positives)
    if total_gt > 0:
        metrics['recall'] = true_positives / total_gt
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['mAP'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    
    return metrics

def visualize_detections(image: np.ndarray,
                        boxes: np.ndarray,
                        scores: np.ndarray = None,
                        class_ids: np.ndarray = None) -> np.ndarray:
    """
    Visualize detection results on an image.
    
    Args:
        image (np.ndarray): Input image
        boxes (np.ndarray): Detected boxes
        scores (np.ndarray): Detection scores
        class_ids (np.ndarray): Class IDs for each detection
        
    Returns:
        np.ndarray: Image with visualized detections
    """
    import cv2
    
    # Make a copy of the image
    vis_image = image.copy()
    
    # Draw each box
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw rectangle
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add score if available
        if scores is not None:
            score = scores[i]
            label = f'{score:.2f}'
            if class_ids is not None:
                label = f'Class {class_ids[i]}: {label}'
            cv2.putText(vis_image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return vis_image
