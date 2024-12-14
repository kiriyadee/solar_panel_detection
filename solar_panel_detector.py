import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D
from tensorflow.keras.models import Model

def create_detection_model():
    # Use ResNet50 as base model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(416, 416, 3))
    
    # Add custom layers for detection
    x = base_model.output
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    # Output: [confidence, x, y, width, height]
    predictions = Conv2D(5, (1, 1), activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def preprocess_image(image_path):
    # Read and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img.shape[:2]
    img_resized = cv2.resize(img, (416, 416))
    img_normalized = img_resized / 255.0
    return img, img_normalized, original_size

def detect_solar_panels(image_path, model, confidence_threshold=0.5):
    # Load and preprocess image
    original_img, processed_img, original_size = preprocess_image(image_path)
    
    # Get predictions
    predictions = model.predict(np.expand_dims(processed_img, axis=0))[0]
    
    # Extract bounding boxes
    boxes = []
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            confidence = predictions[i, j, 0]
            if confidence > confidence_threshold:
                # Get box coordinates
                x = predictions[i, j, 1] * original_size[1]
                y = predictions[i, j, 2] * original_size[0]
                w = predictions[i, j, 3] * original_size[1]
                h = predictions[i, j, 4] * original_size[0]
                
                boxes.append({
                    'confidence': confidence,
                    'x': int(x - w/2),
                    'y': int(y - h/2),
                    'w': int(w),
                    'h': int(h)
                })
    
    return original_img, boxes

def draw_boxes(image, boxes):
    img_with_boxes = image.copy()
    for box in boxes:
        # Draw rectangle
        cv2.rectangle(img_with_boxes, 
                     (box['x'], box['y']), 
                     (box['x'] + box['w'], box['y'] + box['h']),
                     (0, 255, 0), 2)
        
        # Add confidence score
        label = f"{box['confidence']:.2f}"
        cv2.putText(img_with_boxes, label, 
                    (box['x'], box['y'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_with_boxes

def detect_and_mark_panels(image_path):
    # Create and compile model
    model = create_detection_model()
    model.compile(optimizer='adam', loss='mse')
    
    # Detect solar panels
    image, boxes = detect_solar_panels(image_path, model)
    
    # Draw boxes on image
    marked_image = draw_boxes(image, boxes)
    
    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(marked_image)
    plt.title('Detected Solar Panels')
    plt.axis('off')
    
    plt.show()
    
    return marked_image

if __name__ == "__main__":
    # Example usage
    image_path = "your_image.jpg"  # Replace with your image path
    marked_image = detect_and_mark_panels(image_path)
