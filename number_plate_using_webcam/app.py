import cv2
import easyocr

# Initialize the webcam (use 0 for the default camera)
cap = cv2.VideoCapture(0)

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])  # Specify the language; 'en' is for English.

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame was read successfully
    if not ret:
        print("Failed to grab frame")
        break

    # Perform OCR on the current frame
    results = reader.readtext(frame)

    # Draw bounding boxes and annotate detected text
    for (bbox, text, confidence) in results:
        # Extract the bounding box coordinates
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Draw a rectangle around detected text
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Annotate the detected text
        cv2.putText(frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 0), 2)

    # Display the frame in a window
    cv2.imshow("Webcam OCR", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
