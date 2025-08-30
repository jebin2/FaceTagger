import json
import cv2
import os
import sys
import termios
import tty

# Function to get a single keypress (no Enter needed)
def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

# Path to JSON
json_path = '/home/jebin/git/CaptionCreator/reuse/movie_review_Blue is the Warmest Color 2013/face.json'

# Load JSON
with open(json_path, 'r') as f:
    data = json.load(f)

# Loop through each entry in JSON
for item in data:
    frame_path = item.get('frame_path')
    face_locations = item.get('face_location')

    # Load the image
    if not os.path.exists(frame_path):
        print(f"Image not found: {frame_path}")
        continue
    img = cv2.imread(frame_path)

    # Draw rectangle if face exists
    if face_locations:
        left, top, right, bottom = face_locations
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

    # Save annotated image
    cv2.imwrite("annotated.jpg", img)
    print(f"Annotated image saved: annotated.jpg")

    # Show the image
    
    print("Press 'y' to go to next image, any other key to quit.")
    key = get_key()
    if key.lower() != 'y':
        break

print("Done processing images.")
