import time

import cv2


def test_triple_camera():
    # Initialize three camera captures
    cap1 = cv2.VideoCapture(0)  # First camera
    cap2 = cv2.VideoCapture(0)  # Second camera

    # Check if first two cameras opened successfully
    if not cap1.isOpened():
        print("Error: Could not open first camera")
        return
    if not cap2.isOpened():
        print("Error: Could not open second camera")
        return

    # Try to open third camera, but continue with two if it fails
    cap3 = cv2.VideoCapture(2)  # Third camera
    use_third_camera = cap3.isOpened()

    if use_third_camera:
        print("All three cameras opened successfully")
    else:
        print("Only two cameras opened successfully, third camera unavailable")

    # Set some basic properties
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    if use_third_camera:
        cap3.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    try:
        while True:
            # Read frames from cameras
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if use_third_camera:
                ret3, frame3 = cap3.read()
                if not ret3:
                    print("Error: Could not read frame from third camera")
                    break

            if not ret1 or not ret2:
                print("Error: Could not read frames from one or more cameras")
                break

            # Display frames
            cv2.imshow("index 0", frame1)
            cv2.imshow("index 1", frame2)
            if use_third_camera:
                cv2.imshow("Camera 3", frame3)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Small delay to prevent high CPU usage
            time.sleep(0.01)

    finally:
        # Release resources
        cap1.release()
        cap2.release()
        if use_third_camera:
            cap3.release()
        cv2.destroyAllWindows()
        print("Cameras released and windows closed")


if __name__ == "__main__":
    test_triple_camera()
