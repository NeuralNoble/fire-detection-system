# fire-detection

This repository contains the code for training a fire detection model using YOLOv8Nano on a custom dataset. The model is trained with the Adam optimizer, a learning rate of 0.002, and a batch size of 25 to accommodate a large dataset. The training was conducted for 50 epochs to ensure the model learns effectively.

## Training
The model was trained using the YOLOv8Nano architecture, which is a lightweight version of YOLOv8 designed for resource-constrained environments. The Adam optimizer was chosen for its efficiency in handling large datasets, and a learning rate of 0.002 was selected based on empirical testing to achieve optimal performance. The batch size of 25 was chosen to balance between training speed and memory usage, given the large size of the dataset. Training was conducted for 50 epochs to ensure the model converged to a stable state

## Model Summary
### Model Architecture: YOLOv8Nano
- Optimizer: Adam
- Learning Rate: 0.002
- Batch Size: 25
- Epochs: 50
- Dataset: Custom dataset containing images of fire scenarios
- Annotations: Bounding box annotations around fire objects
### Metrics:
- Precision (B): 0.577
- Recall (B): 0.426
- mAP50 (B): 0.495
- mAP50-95 (B): 0.224
### Challenges:
- Dataset size was relatively low i have plans to use bigger datset for improving the model
- Bounding boxes were inaccurately marked
- Loss for bounding boxes reached 1.2 after 50 epochs, indicating difficulty in convergence 

### Model Quantization with TensorFlow Lite
After training my YOLOv8Nano model, I quantized it to reduce its size and make it more suitable for deployment on resource-constrained devices. TensorFlow Lite (TFLite) provides tools to convert your trained model to a TFLite-compatible format. after quantization the model size got down to 5.3 mb
