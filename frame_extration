import cv2
import os

video_path = "videos/GX012517.MP4"
output_folder = "frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS:", fps)

frame_number = 0
saved = 0

while True:
    cap.set(cv2.CAP_PROP_POS_MSEC, frame_number * 1000)

    ret, frame = cap.read()

    if not ret:
        break

    filename = os.path.join(
        output_folder,
        f"frame_{saved:04d}.jpg"
    )

    cv2.imwrite(filename, frame)

    saved += 1
    frame_number += 1

cap.release()

print(f"{saved} frames extracted successfully!")


-----------------------------------------------------------------------------------------------
