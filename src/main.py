import cv2
import pytesseract
from ultralytics import YOLO
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# ========================
# Предобработка
# ========================
def puttext_ru(img, text, pos, font_path="arial.ttf", font_size=32, color=(0,255,0)):
    # OpenCV -> PIL
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(pos, text, font=font, fill=color)
    # PIL -> OpenCV
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

def preprocess_plate(plate_img):
    """Улучшение читаемости перед OCR."""
    plate = cv2.resize(plate_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    return gray

def preprocess_plate(plate_img):
    plate = cv2.resize(plate_img, None, fx=2.0, fy=2.0,
                       interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # выравнивание контрастности
    gray = cv2.equalizeHist(gray)

    # адаптивная бинаризация
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 3
    )
    return bin_img


# ========================
# Фильтрация мусора OCR
# ========================
def is_valid_plate(text: str) -> bool:
    """Отбрасываем мусорные распознавания OCR."""
    if len(text) < 5 or len(text) > 10:
        return False
    
    # Должны быть и буквы, и цифры
    has_letter = any(ch.isalpha() for ch in text)
    has_digit = any(ch.isdigit() for ch in text)
    if not (has_letter and has_digit):
        return False
    
    # Слишком много одинаковых символов
    if text.count(text[0]) > len(text) * 0.7:
        return False
    
    return True


# ========================
# OCR
# ========================
def extract_plate_text(plate_img):
    """Распознавание номера."""
    proc = preprocess_plate(plate_img)
    text = pytesseract.image_to_string(proc, config=config)
    text = ''.join(ch for ch in text if ch.isalnum())
    return text


# ========================
# MAIN PIPELINE
# ========================
def main():
    cap = cv2.VideoCapture("../data/videoplayback.mp4")

    if not cap.isOpened():
        print("Не удалось открыть видеопоток")
        return

    # ================================
    # Настройка видеозаписи
    # ================================
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        "../data/detected.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO детекция
        results = model.predict(frame, verbose=False)[0]

        for box in results.boxes:
            cls = int(box.cls[0])

            # Номерной знак = класс 0
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate = frame[y1:y2, x1:x2]

                # OCR
                text = extract_plate_text(plate)
                
                if is_valid_plate(text):
                    print("Распознан номер:", text)

                # Рисуем прямоугольник
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)

        # Показываем
        cv2.imshow("License Plate Recognition", frame)

        # Записываем в файл
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# ========================
# RUN
# ========================
if __name__ == "__main__":
    model = YOLO("../weights/best.pt")
    config = (
        "-l rus "
        "-c tessedit_char_whitelist=АВЕКМНОРСТУХ0123456789"
        "--psm 8 "
        "--oem 3"
    )
    main()
