import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        self.image = None
        self.contours = None
        self.contour_points = None

    def process_image(self, image_path):
        """Обработка изображения и извлечение контуров"""
        # Загрузка изображения
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Не удалось загрузить изображение")

        # Преобразование в градации серого
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Размытие для уменьшения шума
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Бинаризация изображения для лучшего выделения контуров
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        # Обнаружение краев
        edges = cv2.Canny(binary, 50, 150)

        # Нахождение контуров
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Выбор самого большого контура
        if not contours:
            raise ValueError("Контуры не найдены")

        main_contour = max(contours, key=cv2.contourArea)
        self.contours = [main_contour]

        # Упрощение контура для уменьшения количества точек
        epsilon = 0.005 * cv2.arcLength(main_contour, True)
        approx_contour = cv2.approxPolyDP(main_contour, epsilon, True)

        # Преобразование точек контура в 3D координаты
        points_3d = self.contour_to_3d(approx_contour)

        self.contour_points = np.array(points_3d)
        return self.contour_points

    def contour_to_3d(self, contour):
        """Преобразование 2D контура в 3D точки с равномерным распределением"""
        height, width = self.image.shape[:2]
        scale = 100.0  # Масштаб для 3D пространства

        # Вычисляем центр изображения
        center_x = width / 2
        center_y = height / 2

        # Преобразуем контур в массив точек
        points_2d = []
        for point in contour:
            points_2d.append([point[0][0], point[0][1]])
        points_2d = np.array(points_2d)

        # Вычисляем центр контура
        contour_center = np.mean(points_2d, axis=0)

        # Нормализуем точки относительно центра изображения
        normalized_points = []
        for point in points_2d:
            x = (point[0] - center_x) * scale / width
            y = (point[1] - center_y) * scale / height
            normalized_points.append([x, y])

        # Преобразуем в 3D точки (x, y, z), где z - высота над поверхностью
        points_3d = []
        for point in normalized_points:
            # Инвертируем y для правильной ориентации в 3D
            # x - восток, y - высота (в данном случае постоянная), z - север
            points_3d.append(np.array([point[0], 0, -point[1]]))

        return points_3d

    def distribute_points_evenly(self, num_points):
        """Равномерно распределяет указанное количество точек по контуру"""
        if self.contour_points is None or len(self.contour_points) < 2:
            raise ValueError("Контур не определен или слишком короткий")

        # Вычисляем периметр контура (приближенно, используя евклидово расстояние)
        perimeter = 0
        for i in range(len(self.contour_points)):
            next_idx = (i + 1) % len(self.contour_points)
            perimeter += np.linalg.norm(self.contour_points[next_idx] - self.contour_points[i])

        # Шаг между точками
        step = perimeter / num_points

        # Равномерно распределяем точки
        result = []
        current_dist = 0
        current_idx = 0
        result.append(self.contour_points[0].copy())

        for _ in range(1, num_points):
            remaining_dist = step
            while True:
                next_idx = (current_idx + 1) % len(self.contour_points)
                segment_length = np.linalg.norm(self.contour_points[next_idx] - self.contour_points[current_idx])

                if current_dist + segment_length >= remaining_dist:
                    # Интерполируем точку на сегменте
                    t = (remaining_dist - current_dist) / segment_length
                    new_point = self.contour_points[current_idx] + t * (self.contour_points[next_idx] - self.contour_points[current_idx])
                    result.append(new_point)
                    current_dist = 0
                    current_idx = next_idx if t == 1 else current_idx
                    break
                else:
                    remaining_dist -= (segment_length - current_dist)
                    current_dist = 0
                    current_idx = next_idx

        return np.array(result)

    def get_recommended_drone_count(self):
        """Рекомендуемое число дронов равно точкам"""
        if self.contour_points is None:
            return 0

        return len(self.contour_points)