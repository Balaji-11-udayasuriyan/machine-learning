import cv2
import easyocr

# Load an image_path
image_path = cv2.imread("D:\\vechile\\vechilenumber.jpg")

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])  # Specify the language; 'en' is for English.

# Perform OCR on the image_path
results = reader.readtext(image_path)

# Draw bounding boxes and annotate detected text
for (bbox, text, confidence) in results:
    # Extract the bounding box coordinates
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))

    # Draw a rectangle around detected text
    cv2.rectangle(image_path, top_left, bottom_right, (0, 255, 0), 2)

    # Annotate the detected text
    cv2.putText(image_path, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 0, 0), 2)

# Display the image_path in a window
cv2.imshow("image_path", image_path)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()