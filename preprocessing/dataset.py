"""
Dataset and data loading utilities for solar panel detection.
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Tuple
from .transforms import ImageTransform

class SolarPanelDataset(Dataset):
    """Dataset class for solar panel detection."""
    
    def __init__(self, 
                 image_dir: str,
                 annotation_file: str = None,
                 transform: ImageTransform = None,
                 train: bool = True):
        """
        Initialize the dataset.
        
        Args:
            image_dir (str): Directory containing images
            annotation_file (str): Path to annotation file
            transform (ImageTransform): Image transformation pipeline
            train (bool): Whether this is training set
        """
        self.image_dir = image_dir
        self.transform = transform or ImageTransform()
        self.train = train
        
        self.images = []
        self.annotations = {}
        
        self._load_dataset(annotation_file)
        
    def _load_dataset(self, annotation_file: str):
        """Load dataset images and annotations."""
        if annotation_file and os.path.exists(annotation_file):
            # Load annotations if file exists
            # Format: image_name, x1, y1, x2, y2, class_id
            with open(annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    image_name = parts[0]
                    bbox = [float(x) for x in parts[1:5]]
                    class_id = int(parts[5])
                    
                    if image_name not in self.annotations:
                        self.annotations[image_name] = []
                    self.annotations[image_name].append({
                        'bbox': bbox,
                        'class_id': class_id
                    })
                    
                    if image_name not in self.images:
                        self.images.append(image_name)
        else:
            # If no annotation file, just load all images
            for img_file in os.listdir(self.image_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(img_file)
                    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.images)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Dictionary containing image and annotation data
        """
        image_name = self.images[idx]
        image_path = os.path.join(self.image_dir, image_name)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Apply transformations
        transformed = self.transform(image)
        
        sample = {
            'image': transformed['image'],
            'image_id': image_name
        }
        
        # Add annotations if available
        if image_name in self.annotations:
            sample['annotations'] = self.annotations[image_name]
            
        return sample
