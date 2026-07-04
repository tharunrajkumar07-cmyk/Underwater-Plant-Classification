import cv2
import os
import numpy as np

# Input folder
input_folder = "frames"

# Output folder
output_folder = "enhanced_frames_v2"

os.makedirs(output_folder, exist_ok=True)

def enhance_underwater(img):

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    # CLAHE enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8,8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l,a,b))

    img = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    # Denoising
    img = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        5,
        5,
        7,
        21
    )

    # Sharpening
    kernel = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ])

    img = cv2.filter2D(
        img,
        -1,
        kernel
    )

    return img

count = 0

for file in os.listdir(input_folder):

    if file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):

        path = os.path.join(
            input_folder,
            file
        )

        img = cv2.imread(path)

        if img is None:
            continue

        enhanced = enhance_underwater(img)

        save_path = os.path.join(
            output_folder,
            file
        )

        cv2.imwrite(
            save_path,
            enhanced
        )

        count += 1

        print("Processed:", file)

print("\nDone!")
print("Total Images Processed:", count)
print("Saved in:", output_folder)
