from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt"):
        # Load the model
        self.model = YOLO(model_name)

    def detect(self, frame, conf_threshold=0.5):
        # Run inference on the frame
        results = self.model(frame, conf=conf_threshold)
        return results[0]