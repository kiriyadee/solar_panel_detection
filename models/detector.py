"""
Base detector class and implementations of various object detection models
for solar panel detection.
"""

import torch
import torch.nn as nn

class BaseDetector(nn.Module):
    """Base class for all solar panel detectors."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            dict: Dictionary containing detection results including boxes, scores, and classes
        """
        raise NotImplementedError
        
    def predict(self, image):
        """
        Perform prediction on a single image.
        
        Args:
            image (numpy.ndarray): Input image in RGB format
            
        Returns:
            dict: Dictionary containing detection results
        """
        raise NotImplementedError
