import cv2
from detector import ObjectDetector

def main():
    # 1. Initialize the engine
    detector = ObjectDetector()
    
    # 2. Access the webcam
    cap = cv2.VideoCapture(0)

    print("System running. Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 3. Use the engine to detect objects
        result = detector.detect(frame)

        # 4. Draw results on the frame
        annotated_frame = result.plot()

        # 5. Display
        cv2.imshow("VisionTrack System", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()