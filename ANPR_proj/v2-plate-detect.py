import cv2
import numpy as np
import os
import pytesseract
import re
from ultralytics import YOLO

script_dir = os.path.dirname(os.path.abspath(__file__))

image_list = ["car1.jpg", "car2.jpg", "car3.jpg", "car4.jpg", "car5.jpg"]
model = YOLO(os.path.join(script_dir, "license_plate_detector.pt"))

def validate_plate(text):
    pattern = r"^[A-Z0-9]{4,8}$"
    return "VALID" if re.match(pattern, text) else "INVALID"


for image_name in image_list:

    image_path = os.path.join(script_dir, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(image_name, "not found")
        continue

    image = cv2.resize(image, None, fx=1.5, fy=1.5)

    print("\nProcessing:", image_name)

    results = model(image, conf=0.10, iou=0.4)

    best_box = None
    best_conf = 0

    for r in results:
        for box in r.boxes:

            conf = float(box.conf[0])

            if conf > best_conf:
                best_conf = conf
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                best_box = (x1, y1, x2, y2)

    plate_crop = None

    if best_box is not None:

        x1, y1, x2, y2 = best_box

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        plate_crop = image[y1:y2, x1:x2]

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=0)

        _, gray = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        plate_text = pytesseract.image_to_string(
            gray,
            config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        text = plate_text.replace(" ", "").strip().upper()

        print("OCR RESULT:", text)
        print("VALIDATION:", validate_plate(text))

    else:
        print("No plate detected in", image_name)

    cv2.imshow("ANPR Detection", image)

    if plate_crop is not None:
        cv2.imshow("Plate Crop", plate_crop)


# LIVE CAMERA
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.10, iou=0.4)

    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("LIVE ANPR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()