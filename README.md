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

### Model Deployment with OpenCV
I deployed my quantized YOLOv8Nano model for fire detection on my laptop using OpenCV. Here's an overview of the deployment process:
- Model Loading: I loaded the quantized YOLOv8Nano model using the Ultralytics library and moved it to the available device (CPU or GPU).

- Video Stream Processing: I captured frames from a video stream using OpenCV and passed each frame through the model for inference.

- Result Visualization: For each detected object, I drew a rectangle around it and displayed the class name and confidence score on the frame using CVZone.

## Future Plans
In future iterations, I plan to enhance the accuracy and reliability of the fire detection system by adding another layer to the workflow. Here's an outline of the proposed improvement:

- Drone-Based Detection: The drone will continue to detect fire in its vicinity using the deployed YOLOv8Nano model.

- Sending Frames to Central Hub: Upon detecting a fire, the drone will send the corresponding frame to a central hub for further analysis.

- Central Hub Processing: At the central hub, a custom-trained ResNet or VGG model will classify the detection as a false alarm or a positive alarm. This step will help reduce false positives and improve the overall accuracy of the system.

- Sending Coordinates and Image: If the detection is classified as a positive alarm, the drone hub will send the coordinates of the fire and the corresponding image frame for further action or alerting authorities.
