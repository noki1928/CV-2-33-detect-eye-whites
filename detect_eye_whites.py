import cv2
import numpy as np
import os


def load_image(path: str):
    """
    Загружает изображение по указанному пути.

    Args:
        path (str): путь к изображению

    Returns:
        np.ndarray: изображение в формате BGR

    Raises:
        FileNotFoundError: если файл не найден или не читается
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл '{path}' не найден.")
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return image


def detect_faces(image: np.ndarray, scale_factor: float = 1.3, min_neighbors: int = 5):
    """
    Находит лица на изображении с помощью каскада Хаара.

    Args:
        image (np.ndarray): изображение BGR
        scale_factor (float): коэффициент масштабирования
        min_neighbors (int): минимальное количество соседей

    Returns:
        list[tuple[int, int, int, int]]: список координат (x, y, w, h)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
    return faces


def detect_eyes(image: np.ndarray, scale_factor: float = 1.3, min_neighbors: int = 6):
    """
    Находит глаза на изображении с помощью каскада Хаара.
    Обычно используется внутри области лица.

    Args:
        image (np.ndarray): изображение лица (BGR)
        scale_factor (float): коэффициент масштабирования
        min_neighbors (int): минимальное количество соседей

    Returns:
        list[tuple[int, int, int, int]]: список координат глаз (x, y, w, h)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes = eye_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
    return eyes


def detect_eye_whites(image: np.ndarray, eyes: list[tuple[int, int, int, int]], step: int = 10):
    """
    Находит и подсвечивает белки глаз на изображении.

    Args:
        image (np.ndarray): изображение (BGR)
        eyes (list): список координат найденных глаз
        step (int): шаг увеличения порога яркости

    Returns:
        tuple[np.ndarray, int]: (обновлённое изображение, количество найденных белков)
    """
    total_whites = 0
    b_channel = cv2.split(image)[0]  # используем синий канал

    for (x, y, w, h) in eyes:
        # Центральная часть глаза
        x1, y1 = x + int(w * 0.2), y + int(h * 0.2)
        x2, y2 = x + int(w * 0.8), y + int(h * 0.8)

        roi = b_channel[y1:y2, x1:x2]
        roi_color = image[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        mid = np.median(roi)
        valid_contours = []

        # Подбор порога по количеству найденных областей
        while mid < 255:
            _, mask = cv2.threshold(roi, float(mid), 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if 20 < cv2.contourArea(c) < 500]

            if len(valid_contours) <= 1:
                break
            mid += step

        cv2.drawContours(roi_color, valid_contours, -1, (0, 255, 255), 2)
        total_whites += len(valid_contours)

    return image, total_whites


def main(image_path: str = "image.jpg"):
    """
    Главная функция: ищет лицо, затем глаза внутри него, и белки глаз.
    """
    try:
        image = load_image(image_path)
    except FileNotFoundError as e:
        print(e)
        return

    faces = detect_faces(image)
    if len(faces) == 0:
        print("Лицо не найдено.")
        return

    total_whites = 0
    total_eyes = 0

    for (x, y, w, h) in faces:
        face_roi = image[y:y + h, x:x + w]
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        eyes = detect_eyes(face_roi)
        total_eyes += len(eyes)

        if len(eyes) == 0:
            continue

        processed_face, whites = detect_eye_whites(face_roi, eyes)
        total_whites += whites
        image[y:y + h, x:x + w] = processed_face

    print(f"Найдено лиц: {len(faces)}")
    print(f"Найдено глаз: {total_eyes}")
    print(f"Найдено белков: {total_whites}")

    cv2.putText(image, f'Whites: {total_whites}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Detected Faces, Eyes, and Eye Whites', image)
    
    try:
        cv2.imwrite('output.jpg', image)
    except Exception as e:
        print(f"Не удалось сохранить изображение: {e}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
