import cv2
from detector import ObjectDetector

def get_camera():
    # Dynamically probes for an available camera index (0-3).
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Successfully connected to camera index {i}")
            return cap
    raise Exception("No camera found! Please check your hardware connections.")

def main():
    # 1. Initialize the engine
    detector = ObjectDetector()
    
    # 2. Access the webcam using the hardware abstraction layer
    try:
        cap = get_camera()
    except Exception as e:
        print(f"Error: {e}")
        return

    print("System running. Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame. Exiting...")
            break

        # 3. Use the engine to detect objects
        result = detector.detect(frame)

        # 4. Draw results on the frame
        annotated_frame = result.plot()

        # 5. Display the feed
        cv2.imshow("VisionTrack System", annotated_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()