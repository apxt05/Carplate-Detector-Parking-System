import cv2
import numpy as np
import os
import pytesseract
import re


script_dir = os.path.dirname(os.path.abspath(__file__))

image_list = ["car1.jpg", "car2.jpg", "car3.jpg", "car4.jpg"]

for image_name in image_list:

    image_path = os.path.join(script_dir, image_name)
    image = cv2.imread(image_path)
    

    if image is None:
        print(image_name, "not found")
        continue

    print("\nProcessing:", image_name)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.bilateralFilter(gray, 11, 17, 17)

    edges = cv2.Canny(blur, 30, 200)

    contours, _ = cv2.findContours(
        edges.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    plate_crop = None
    best_box = None
    best_ratio_score = float('inf')


    for cnt in contours:

        approx = cv2.approxPolyDP(
            cnt,
            0.02 * cv2.arcLength(cnt, True),
            True
        )

        if len(approx) == 4:

            x, y, w, h = cv2.boundingRect(cnt)

            aspect_ratio = w / float(h)

            if 2 < aspect_ratio < 6:

                ratio_score = abs(4 - aspect_ratio)

                if ratio_score < best_ratio_score or best_box is None:
                    best_ratio_score = ratio_score
                    best_box = (x, y, w, h)


    if best_box is not None:

        x, y, w, h = best_box

        plates = [(x, y, w, h)]
        for (x, y, w, h) in plates:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        plate_crop = image[y:y+h, x:x+w]

        # task 4 - annotation output
        os.makedirs(os.path.join(script_dir, "output"), exist_ok=True)

        cv2.imwrite(
            os.path.join(script_dir, f"output/detected_{image_name}"),
            image
        )

        # task 3 - OCR output
        plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

        # improve contrast first
        plate_gray = cv2.convertScaleAbs(plate_gray, alpha=1.5, beta=0)

        #denoise
        plate_gray = cv2.GaussianBlur(plate_gray, (3,3), 0)

        # threshold
        plate_gray = cv2.threshold(
            plate_gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        plate_text = pytesseract.image_to_string(
            plate_gray,
            config="--psm 7"
        )

        text = plate_text.replace(" ", "").strip()

        print("OCR RESULT:", text)

        # task 7 - validation
        def validate_plate(text):
            pattern = r"^[A-Z]{1,3}[0-9]{1,4}[A-Z]?$"
            if re.match(pattern, text):
                return "VALID"
            else:
                return "INVALID"

        print("VALIDATION:", validate_plate(text))

    else:
        print("No plate detected in", image_name)

    cv2.imshow("ANPR Detection", image)

    if plate_crop is not None:
        cv2.imshow("Plate Crop", plate_crop)


# TASK 5 - real time camera system
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("LIVE ANPR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.waitKey(0)
cv2.destroyAllWindows()