import cv2
import os

# Map gestures to key presses

gesture_mapping = {
    'hi': ord('h'),
    'victory': ord('v'),
    'thumbs_up': ord('t'),
    'V': ord('V'),
    'W': ord('W'),
    'C': ord('C'),
    'L': ord('L'),
    'help': ord('e'),
    'please': ord('p')
}

# Create directories for each gesture
base_path = "C:/Users/jithe/OneDrive/Desktop/asp/maindataset"
gesture_paths = {gesture: os.path.join(base_path, gesture) for gesture in gesture_mapping}
for path in gesture_paths.values():
    os.makedirs(path, exist_ok=True)

# Open a connection to the webcam (assuming it's the first camera, you can change the index if needed)
cap = cv2.VideoCapture(0)

# Create a window to display the captured frame
cv2.namedWindow("Capture Frame")

# Variables to track the number of images captured for each gesture
gesture_counts = {gesture: 0 for gesture in gesture_mapping}

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()

    # Display the captured frame
    cv2.imshow("Capture Frame", frame)

    # Prompt user for input ('q' to quit)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        # Quit the loop
        break

    # Check if the key corresponds to a mapped gesture
    for gesture, gesture_key in gesture_mapping.items():
        if key == gesture_key:
            # Save the frame as an image for the corresponding gesture
            gesture_counts[gesture] += 1
            image_path = os.path.join(gesture_paths[gesture], f'{gesture}_{gesture_counts[gesture]}.png')
            cv2.imwrite(image_path, frame)
            print(f"{gesture.capitalize()} image {gesture_counts[gesture]} saved.")
