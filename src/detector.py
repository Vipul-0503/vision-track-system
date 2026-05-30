# src/detector.py
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)
        # Using exact IDs from the list: 0, 56, 63, 67, 73
        self.target_classes = [0, 56, 63, 67, 73] 

    def detect(self, frame):
        # conf=0.7 forces the model to be 70% sure before drawing a box
        # This eliminates most 'false positive' misclassifications
        results = self.model(frame, classes=self.target_classes, conf=0.7)
        return results[0]