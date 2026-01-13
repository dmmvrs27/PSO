import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

class Visualization3D(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill=tk.BOTH, expand=True)

        # Инициализация параметров
        self.width = 800
        self.height = 600
        self.fov = 45
        self.near = 0.1
        self.far = 1000.0

        # Базовые географические координаты
        self.base_latitude = 55.7558
        self.base_longitude = 37.6173
        self.base_altitude = 0

        # Масштабные коэффициенты для преобразования координат
        self.meters_per_degree_lat = 111319.9  # метров в одном градусе широты
        self.meters_per_degree_lon = 111319.9 * np.cos(np.radians(self.base_latitude))

        self.target_points = []
        self.drones = []
        self.animation_id = None
        self.drone_swarm = None

        # Параметры для перетаскивания
        self.dragging = False
        self.selected_drone = None
        self.mouse_pos = None

        # Создание холста OpenGL
        self.create_gl_canvas()

        # Инициализация камеры
        self.camera_distance = 200
        self.camera_rotation = [45, 45, 0]
        self.init_gl()

        # Создание рамки для координат
        self.create_coords_frame()

    def create_coords_frame(self):
        """Создание рамки с информацией о координатах"""
        # Создаем рамку
        self.coords_frame = ttk.LabelFrame(self, text="Информация о дроне", padding="10")
        self.coords_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # Создаем и размещаем метки для каждого параметра
        # Добавляем поле ID дрона
        self.id_frame = ttk.Frame(self.coords_frame)
        self.id_frame.pack(fill=tk.X, pady=5)

        self.id_label = ttk.Label(self.id_frame, text="ID дрона:", font=("Arial", 10, "bold"))
        self.id_label.pack(side=tk.LEFT, padx=5)
        self.id_value = ttk.Label(self.id_frame, text="N/A", font=("Arial", 10))
        self.id_value.pack(side=tk.LEFT, padx=5)

        # Географические координаты
        self.geo_frame = ttk.Frame(self.coords_frame)
        self.geo_frame.pack(fill=tk.X, expand=True)

        # Левая колонка - координаты
        self.left_frame = ttk.Frame(self.geo_frame)
        self.left_frame.pack(side=tk.LEFT, padx=10)

        self.lat_label = ttk.Label(self.left_frame, text="Широта:", font=("Arial", 10, "bold"))
        self.lat_label.pack(anchor=tk.W)
        self.lat_value = ttk.Label(self.left_frame, text="N/A", font=("Arial", 10))
        self.lat_value.pack(anchor=tk.W)

        self.lon_label = ttk.Label(self.left_frame, text="Долгота:", font=("Arial", 10, "bold"))
        self.lon_label.pack(anchor=tk.W)
        self.lon_value = ttk.Label(self.left_frame, text="N/A", font=("Arial", 10))
        self.lon_value.pack(anchor=tk.W)

        # Центральная колонка - высота и скорость
        self.center_frame = ttk.Frame(self.geo_frame)
        self.center_frame.pack(side=tk.LEFT, padx=10)

        self.alt_label = ttk.Label(self.center_frame, text="Высота (м):", font=("Arial", 10, "bold"))
        self.alt_label.pack(anchor=tk.W)
        self.alt_value = ttk.Label(self.center_frame, text="N/A", font=("Arial", 10))
        self.alt_value.pack(anchor=tk.W)

        self.speed_label = ttk.Label(self.center_frame, text="Скорость (м/с):", font=("Arial", 10, "bold"))
        self.speed_label.pack(anchor=tk.W)
        self.speed_value = ttk.Label(self.center_frame, text="N/A", font=("Arial", 10))
        self.speed_value.pack(anchor=tk.W)

        # Правая колонка - дополнительная информация
        self.right_frame = ttk.Frame(self.geo_frame)
        self.right_frame.pack(side=tk.LEFT, padx=10)

        self.heading_label = ttk.Label(self.right_frame, text="Курс (°):", font=("Arial", 10, "bold"))
        self.heading_label.pack(anchor=tk.W)
        self.heading_value = ttk.Label(self.right_frame, text="N/A", font=("Arial", 10))
        self.heading_value.pack(anchor=tk.W)

        self.distance_label = ttk.Label(self.right_frame, text="До цели (м):", font=("Arial", 10, "bold"))
        self.distance_label.pack(anchor=tk.W)
        self.distance_value = ttk.Label(self.right_frame, text="N/A", font=("Arial", 10))
        self.distance_value.pack(anchor=tk.W)

        # Локальные 3D координаты
        self.local_frame = ttk.LabelFrame(self.coords_frame, text="Локальные 3D координаты", padding="5")
        self.local_frame.pack(fill=tk.X, pady=5)

        self.local_coords_frame = ttk.Frame(self.local_frame)
        self.local_coords_frame.pack(fill=tk.X)

        # X координата
        self.x_label = ttk.Label(self.local_coords_frame, text="X:", font=("Arial", 10, "bold"))
        self.x_label.grid(row=0, column=0, padx=5)
        self.x_value = ttk.Label(self.local_coords_frame, text="N/A", font=("Arial", 10))
        self.x_value.grid(row=0, column=1, padx=5)

        # Y координата
        self.y_label = ttk.Label(self.local_coords_frame, text="Y:", font=("Arial", 10, "bold"))
        self.y_label.grid(row=0, column=2, padx=5)
        self.y_value = ttk.Label(self.local_coords_frame, text="N/A", font=("Arial", 10))
        self.y_value.grid(row=0, column=3, padx=5)

        # Z координата
        self.z_label = ttk.Label(self.local_coords_frame, text="Z:", font=("Arial", 10, "bold"))
        self.z_label.grid(row=0, column=4, padx=5)
        self.z_value = ttk.Label(self.local_coords_frame, text="N/A", font=("Arial", 10))
        self.z_value.grid(row=0, column=5, padx=5)

        # Скорость по компонентам
        self.velocity_frame = ttk.LabelFrame(self.coords_frame, text="Компоненты скорости (м/с)", padding="5")
        self.velocity_frame.pack(fill=tk.X, pady=5)

        self.velocity_coords_frame = ttk.Frame(self.velocity_frame)
        self.velocity_coords_frame.pack(fill=tk.X)

        # Скорость по X
        self.vx_label = ttk.Label(self.velocity_coords_frame, text="Vx:", font=("Arial", 10, "bold"))
        self.vx_label.grid(row=0, column=0, padx=5)
        self.vx_value = ttk.Label(self.velocity_coords_frame, text="N/A", font=("Arial", 10))
        self.vx_value.grid(row=0, column=1, padx=5)

        # Скорость по Y
        self.vy_label = ttk.Label(self.velocity_coords_frame, text="Vy:", font=("Arial", 10, "bold"))
        self.vy_label.grid(row=0, column=2, padx=5)
        self.vy_value = ttk.Label(self.velocity_coords_frame, text="N/A", font=("Arial", 10))
        self.vy_value.grid(row=0, column=3, padx=5)

        # Скорость по Z
        self.vz_label = ttk.Label(self.velocity_coords_frame, text="Vz:", font=("Arial", 10, "bold"))
        self.vz_label.grid(row=0, column=4, padx=5)
        self.vz_value = ttk.Label(self.velocity_coords_frame, text="N/A", font=("Arial", 10))
        self.vz_value.grid(row=0, column=5, padx=5)

    def local_to_geo(self, local_pos):
        """Преобразование локальных координат в географические"""
        lat_change = local_pos[2] / self.meters_per_degree_lat
        lon_change = local_pos[0] / self.meters_per_degree_lon
        altitude = local_pos[1] + self.base_altitude

        latitude = self.base_latitude + lat_change
        longitude = self.base_longitude + lon_change

        return latitude, longitude, altitude

    def calculate_heading(self, drone_pos, target_pos):
        """Расчет курса дрона"""
        dx = target_pos[0] - drone_pos[0]
        dz = target_pos[2] - drone_pos[2]
        heading = np.degrees(np.arctan2(dx, dz))
        return (heading + 360) % 360

    def calculate_speed(self, velocity):
        """Расчет скорости дрона"""
        if velocity is not None:
            return np.linalg.norm(velocity)
        return 0

    def update_coords_display(self, drone_idx):
        """Обновление отображения координат"""
        if drone_idx is None or self.drone_swarm is None:
            # Сброс всех значений
            for label in [self.id_value, self.lat_value, self.lon_value, self.alt_value,
                          self.speed_value, self.heading_value, self.distance_value,
                          self.x_value, self.y_value, self.z_value,
                          self.vx_value, self.vy_value, self.vz_value]:
                label.config(text="N/A")
            return

        # Получение информации о дроне
        drone_info = self.drone_swarm.get_drone_info(drone_idx)
        if drone_info is None:
            return

        # Извлечение данных
        position = drone_info['position']
        velocity = drone_info['velocity']
        target = drone_info['target']
        drone_id = drone_info['id']

        # Преобразование координат
        lat, lon, alt = self.local_to_geo(position)

        # Расчет дополнительных параметров
        heading = self.calculate_heading(position, target)
        speed = self.calculate_speed(velocity)
        distance = np.linalg.norm(target - position)

        # Форматирование строковых значений
        lat_str = f"{lat:.6f}° {'N' if lat >= 0 else 'S'}"
        lon_str = f"{lon:.6f}° {'E' if lon >= 0 else 'W'}"
        alt_str = f"{alt:.1f}"
        speed_str = f"{speed:.2f}"
        heading_str = f"{heading:.1f}"
        distance_str = f"{distance:.2f}"

        # Форматирование локальных координат
        x_str = f"{position[0]:.2f}"
        y_str = f"{position[1]:.2f}"
        z_str = f"{position[2]:.2f}"

        # Форматирование компонентов скорости
        vx_str = f"{velocity[0]:.2f}"
        vy_str = f"{velocity[1]:.2f}"
        vz_str = f"{velocity[2]:.2f}"

        # Обновление меток
        self.id_value.config(text=f"{drone_id}")
        self.lat_value.config(text=lat_str)
        self.lon_value.config(text=lon_str)
        self.alt_value.config(text=alt_str)
        self.speed_value.config(text=speed_str)
        self.heading_value.config(text=heading_str)
        self.distance_value.config(text=distance_str)

        # Обновление локальных координат
        self.x_value.config(text=x_str)
        self.y_value.config(text=y_str)
        self.z_value.config(text=z_str)

        # Обновление компонентов скорости
        self.vx_value.config(text=vx_str)
        self.vy_value.config(text=vy_str)
        self.vz_value.config(text=vz_str)

    def create_gl_canvas(self):
        """Создание контекста OpenGL"""
        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Симуляция роя дронов")

    def init_gl(self):
        """Инициализация OpenGL"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)

        # Настройка проекции
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, self.width/self.height, self.near, self.far)

        # Настройка света
        glLightfv(GL_LIGHT0, GL_POSITION, (100, 100, 100, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))

    def draw_direction_marker(self, position, direction, color):
        """Отрисовка маркера направления"""
        glColor3f(*color)
        glPushMatrix()
        glTranslatef(*position)

        # Отрисовка конуса направления
        glBegin(GL_TRIANGLES)
        glVertex3f(0, 0, 0)
        glVertex3f(-0.5, 0, -1.5)
        glVertex3f(0.5, 0, -1.5)
        glEnd()

        glPopMatrix()

    def set_target_points(self, points):
        """Установка целевых точек"""
        self.target_points = points

    def set_drones(self, drones):
        """Установка позиций дронов"""
        self.drones = drones

    def set_drone_swarm(self, swarm):
        """Установка ссылки на объект DroneSwarm"""
        self.drone_swarm = swarm

    def screen_to_world(self, screen_x, screen_y):
        """Преобразование координат экрана в мировые координаты"""
        viewport = glGetIntegerv(GL_VIEWPORT)
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)

        win_x = float(screen_x)
        win_y = float(viewport[3] - screen_y)
        win_z = glReadPixels(int(win_x), int(win_y), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)[0][0]

        world_pos = gluUnProject(win_x, win_y, win_z, modelview, projection, viewport)
        return np.array(world_pos)

    def find_nearest_drone(self, screen_x, screen_y):
        """Поиск ближайшего дрона к точке на экране"""
        min_dist = float('inf')
        nearest_idx = None

        world_pos = self.screen_to_world(screen_x, screen_y)

        for idx, (drone_pos, _, _) in enumerate(self.drones):
            dist = np.linalg.norm(drone_pos - world_pos)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx

        return nearest_idx if min_dist < 10.0 else None

    def draw_scene(self):
        """Отрисовка всей сцены"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glTranslatef(0, 0, -self.camera_distance)
        glRotatef(self.camera_rotation[0], 1, 0, 0)
        glRotatef(self.camera_rotation[1], 0, 1, 0)
        glRotatef(self.camera_rotation[2], 0, 0, 1)

        self.draw_grid()

        # Отрисовка контура целевых точек
        glColor3f(1.0, 0.0, 0.0)  # Красный цвет для контура

        # Рисуем замкнутую линию контура
        if len(self.target_points) > 1:
            glLineWidth(2.0)
            glBegin(GL_LINE_LOOP)
            for point in self.target_points:
                glVertex3fv(point)
            glEnd()
            glLineWidth(1.0)

        # Отрисовка точек целей
        for point in self.target_points:
            self.draw_sphere(point, 0.8)  # Уменьшенный размер для точек

        # Отрисовка дронов
        for idx, (drone, target, drone_id) in enumerate(self.drones):
            if idx == self.selected_drone:
                glColor3f(1.0, 1.0, 0.0)  # Желтый для выбранного дрона
                self.update_coords_display(idx)
            else:
                glColor3f(0.0, 0.0, 1.0)  # Синий для остальных

            self.draw_sphere(drone, 2.0)

            # Линия к цели
            glBegin(GL_LINES)
            glVertex3fv(drone)
            glVertex3fv(target)
            glEnd()

            # Отображение ID дрона над ним
            self.render_text(drone, str(drone_id))

        # Если нет выбранного дрона, очищаем информацию
        if self.selected_drone is None:
            self.update_coords_display(None)

        pygame.display.flip()

    def draw_grid(self):
        """Отрисовка координатной сетки"""
        glBegin(GL_LINES)

        # Основная сетка (серый цвет)
        glColor3f(0.5, 0.5, 0.5)

        # Горизонтальные линии
        for i in range(-10, 11):
            glVertex3f(-100, 0, i * 10)
            glVertex3f(100, 0, i * 10)
            glVertex3f(i * 10, 0, -100)
            glVertex3f(i * 10, 0, 100)

        glEnd()

        # Оси координат
        glBegin(GL_LINES)
        # Ось X (красная) - восток
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(100, 0, 0)
        # Ось Y (зеленая) - высота
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 100, 0)
        # Ось Z (синяя) - север
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 100)
        glEnd()

        # Отрисовка маркеров направлений
        self.draw_direction_marker([100, 0, 0], "E", (1, 0, 0))  # Восток - красный
        self.draw_direction_marker([0, 100, 0], "H", (0, 1, 0))  # Высота - зеленый
        self.draw_direction_marker([0, 0, 100], "N", (0, 0, 1))  # Север - синий
        self.draw_direction_marker([-100, 0, 0], "W", (1, 0, 0))  # Запад - красный
        self.draw_direction_marker([0, 0, -100], "S", (0, 0, 1))  # Юг - синий

    def draw_sphere(self, position, radius):
        """Отрисовка сферы"""
        glPushMatrix()
        glTranslatef(*position)
        quad = gluNewQuadric()
        gluSphere(quad, radius, 16, 16)
        glPopMatrix()

    def render_text(self, position, text):
        """Отображение текста в 3D пространстве (упрощенная версия)"""
        # Метод не позволяет реально отобразить текст в 3D из-за ограничений OpenGL/Pygame
        # Для полноценной работы необходимо использовать библиотеки для рендеринга текста
        pass

    def start_animation(self):
        """Запуск анимации"""
        if not self.animation_id:
            self.animation_id = self.after(16, self.update)

    def stop_animation(self):
        """Остановка анимации"""
        if self.animation_id:
            self.after_cancel(self.animation_id)
            self.animation_id = None

    def update(self):
        """Обновление сцены"""
        # Обработка событий Pygame
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Левая кнопка мыши
                    # Проверка на выбор дрона
                    self.selected_drone = self.find_nearest_drone(event.pos[0], event.pos[1])
                    if self.selected_drone is not None and self.drone_swarm:
                        self.drone_swarm.start_dragging(self.selected_drone)
                        self.dragging = True
                        self.mouse_pos = event.pos
                        # Обновляем информацию о дроне
                        self.update_coords_display(self.selected_drone)
                elif event.button == 4:  # Колесико мыши вверх
                    self.camera_distance = max(10, self.camera_distance - 10)
                elif event.button == 5:  # Колесико мыши вниз
                    self.camera_distance = min(500, self.camera_distance + 10)
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1 and self.dragging:
                    if self.selected_drone is not None and self.drone_swarm:
                        self.drone_swarm.stop_dragging(self.selected_drone)
                    self.dragging = False
                    self.selected_drone = None
                    self.mouse_pos = None
                    # Очищаем информацию о дроне
                    self.update_coords_display(None)
            elif event.type == MOUSEMOTION:
                if self.dragging and self.selected_drone is not None:
                    # Обновление позиции дрона при перетаскивании
                    new_pos = self.screen_to_world(event.pos[0], event.pos[1])
                    if self.drone_swarm:
                        self.drone_swarm.set_drone_position(self.selected_drone, new_pos)
                        # Обновляем информацию о дроне при перетаскивании
                        self.update_coords_display(self.selected_drone)
                elif event.buttons[2]:  # Правая кнопка мыши для вращения
                    self.camera_rotation[0] += event.rel[1]
                    self.camera_rotation[1] += event.rel[0]
                    # Ограничение углов поворота
                    self.camera_rotation[0] = min(max(self.camera_rotation[0], -90), 90)
            elif event.type == KEYDOWN:
                if event.key == K_r:  # Клавиша R для сброса камеры
                    self.reset_camera()
                elif event.key == K_SPACE:  # Пробел для перераспределения целей
                    if self.drone_swarm:
                        self.drone_swarm.reassign_targets()

        # Обновление позиций дронов
        if self.drone_swarm:
            self.drone_swarm.update_positions()
            self.drones = self.drone_swarm.get_drones()

        # Отрисовка сцены
        self.draw_scene()

        # Планирование следующего обновления
        self.animation_id = self.after(16, self.update)

    def reset_camera(self):
        """Сброс положения камеры к начальным значениям"""
        self.camera_distance = 200
        self.camera_rotation = [45, 45, 0]

    def set_base_coordinates(self, latitude, longitude, altitude=0):
        """Установка базовых географических координат"""
        self.base_latitude = latitude
        self.base_longitude = longitude
        self.base_altitude = altitude
        # Обновление масштабных коэффициентов
        self.meters_per_degree_lon = 111319.9 * np.cos(np.radians(self.base_latitude))

    def cleanup(self):
        """Очистка ресурсов при закрытии"""
        self.stop_animation()
        pygame.quit()