# fire-detection

This repository contains the code for training a fire detection model using YOLOv8Nano on a custom dataset. The model is trained with the Adam optimizer, a learning rate of 0.002, and a batch size of 25 to accommodate a large dataset. The training was conducted for 50 epochs to ensure the model learns effectively.

## Training
The model was trained using the YOLOv8Nano architecture, which is a lightweight version of YOLOv8 designed for resource-constrained environments. The Adam optimizer was chosen for its efficiency in handling large datasets, and a learning rate of 0.002 was selected based on empirical testing to achieve optimal performance. The batch size of 25 was chosen to balance between training speed and memory usage, given the large size of the dataset. Training was conducted for 50 epochs to ensure the model converged to a stable state

## Evaluation
The trained model was evaluated using standard metrics such as precision, recall, and mean Average Precision (mAP) at different Intersection over Union (IoU) thresholds. The model demonstrated strong performance, achieving a precision of 0.577, a recall of 0.426, an mAP50 of 0.495, and an mAP50-95 of 0.224, indicating its effectiveness in detecting fire objects.

## Model Summary
Model Architecture: YOLOv8Nano
Optimizer: Adam
Learning Rate: 0.002
Batch Size: 25
Epochs: 50
Dataset: Custom dataset containing images of fire scenarios
Annotations: Bounding box annotations around fire objects
Metrics:
Precision (B): 0.577
Recall (B): 0.426
mAP50 (B): 0.495
mAP50-95 (B): 0.224
Challenges:
Dataset size was relatively low i have plans to use bigger datset for improving the model
Bounding boxes were inaccurately marked
Loss for bounding boxes reached 1.2 after 50 epochs, indicating difficulty in convergence 
