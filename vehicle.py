import cv2
import numpy as np

def count_cars(video_path):
    net = cv2.dnn.readNet(r"D:\1. APPS\Works\Akash1\Projects\Vehicle_counting.py\yolov3.cfg", r"D:\1. APPS\Works\Akash1\Projects\Vehicle_counting.py\yolov3.weights")
    layer_names = net.getUnconnectedOutLayersNames()

    cap = cv2.VideoCapture(video_path)

    car_count = 0
    frame_skip = 5  # Process every 5th frame
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(layer_names)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if class_id == 2 and confidence > 0.2:
                    car_count += 1

        cv2.putText(frame, f'Car Count: {car_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Car Counting', frame)

        key = cv2.waitKey(30)  # Delay of 30 milliseconds

        # Add this condition to break the loop when the user closes the window
        if key == 27 or key == ord('q'):
            break

    cv2.waitKey(0)  # Wait indefinitely for a key press
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r"D:\1. APPS\Works\Akash1\Projects\Vehicle_counting.py\video.mp4"
    count_cars(video_path)
