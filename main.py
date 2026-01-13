import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import json
from drone_simulation import DroneSwarm
from visualization import Visualization3D
from image_processor import ImageProcessor

class DroneSwarmApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Визуализация роя дронов")
        self.geometry("1200x800")
        
        # Инициализация компонентов
        self.drone_swarm = None
        self.visualization = None
        self.image_processor = ImageProcessor()
        
        # Установка базовых координат
        self.base_latitude = 55.7558
        self.base_longitude = 37.6173
        self.base_altitude = 20
        
        self.create_gui()
        self.create_menu()
        
        # Планирование обновления информации о рое
        self.after(1000, self.update_swarm_info)
        
    def create_menu(self):
        """Создание главного меню"""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        # Меню Файл
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Загрузить изображение", command=self.load_image)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.quit)
        
        # Меню Настройки
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Настройки", menu=settings_menu)
        settings_menu.add_command(label="Задать базовые координаты", 
                                command=self.set_base_coordinates)
        settings_menu.add_command(label="Перераспределить цели",
                                command=self.reassign_targets)
        
        # Меню Вид
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Вид", menu=view_menu)
        view_menu.add_command(label="Сброс камеры", command=self.reset_camera)
        
        # Меню Справка
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Справка", menu=help_menu)
        help_menu.add_command(label="О программе", command=self.show_about)
        help_menu.add_command(label="Инструкция", command=self.show_help)
    
    def update_swarm_info(self):
        """Обновление информации о состоянии роя"""
        if self.drone_swarm:
            error = self.drone_swarm.get_average_error()
            self.error_var.set(f"{error:.2f} м")
            
            if self.drone_swarm.is_converged():
                self.converge_var.set("Сходимость достигнута")
            else:
                self.converge_var.set("В процессе")
        
        # Планирование следующего обновления
        self.after(1000, self.update_swarm_info)
    
    def reset_camera(self):
        """Сброс положения камеры"""
        if self.visualization:
            self.visualization.reset_camera()
    
    def reassign_targets(self):
        """Перераспределение целевых точек между дронами"""
        if self.drone_swarm:
            self.drone_swarm.reassign_targets()
            messagebox.showinfo(
                "Информация",
                "Целевые точки перераспределены между дронами"
            )
        else:
            messagebox.showerror(
                "Ошибка",
                "Сначала необходимо загрузить изображение и создать рой"
            )
    
    def start_simulation(self):
        """Запуск симуляции"""
        self.start_button['state'] = tk.DISABLED
        self.stop_button['state'] = tk.NORMAL
        self.visualization.start_animation()
        
    def stop_simulation(self):
        """Остановка симуляции"""
        self.start_button['state'] = tk.NORMAL
        self.stop_button['state'] = tk.DISABLED
        self.visualization.stop_animation()
    
    def auto_calculate_drones(self):
        """Автоматический расчет рекомендуемого количества дронов"""
        if hasattr(self.image_processor, 'contours') and self.image_processor.contours:
            recommended = self.image_processor.get_recommended_drone_count()
            self.drone_count_var.set(str(recommended))
            messagebox.showinfo(
                "Рекомендация",
                f"Рекомендуемое количество дронов: {recommended}"
            )
        else:
            messagebox.showerror(
                "Ошибка",
                "Сначала необходимо загрузить изображение"
            )
    
    def show_about(self):
        """Показ информации о программе"""
        messagebox.showinfo(
            "О программе",
            "Визуализация роя дронов v1.1\n\n"
            "Программа для моделирования движения роя дронов\n"
            "с использованием алгоритма роевого интеллекта\n\n"
            "Обновлено: автоматическое выстраивание дронов по контуру\n"
            "и усовершенствованное отображение координат"
        )
    
    def show_help(self):
        """Показ справки"""
        help_text = """
        Инструкция по использованию:
        
        1. Загрузите изображение через меню 'Файл' или кнопку
        2. Укажите желаемое количество дронов или используйте автоматический расчет
        3. Нажмите 'Запустить' для начала симуляции
        
        Управление камерой:
        - Левая кнопка мыши: выбор и перетаскивание дронов
        - Правая кнопка мыши: вращение камеры
        - Колесико мыши: масштабирование
        - Клавиша R: сброс положения камеры
        - Пробел: перераспределение целевых точек
        
        В режиме симуляции:
        - Дроны автоматически выстраиваются по контуру изображения
        - При выборе дрона отображается подробная информация о его координатах
        
        Для остановки симуляции нажмите 'Стоп'
        """
        messagebox.showinfo("Справка", help_text)
    
    def create_gui(self):
        """Создание графического интерфейса"""
        # Создание фреймов
        self.control_frame = ttk.Frame(self, padding="5")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.visualization_frame = ttk.Frame(self)
        self.visualization_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Элементы управления
        ttk.Label(self.control_frame, text="Панель управления", font=("Arial", 12, "bold")).pack(pady=5)
        
        # Кнопка загрузки изображения
        self.load_button = ttk.Button(
            self.control_frame, 
            text="Загрузить изображение",
            command=self.load_image
        )
        self.load_button.pack(pady=5)
        
        # Базовые координаты
        coords_frame = ttk.LabelFrame(self.control_frame, text="Базовые координаты", padding="5")
        coords_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(coords_frame, text=f"Широта: {self.base_latitude}° N").pack()
        ttk.Label(coords_frame, text=f"Долгота: {self.base_longitude}° E").pack()
        ttk.Label(coords_frame, text=f"Высота: {self.base_altitude} м").pack()
        
        # Параметры симуляции
        sim_frame = ttk.LabelFrame(self.control_frame, text="Параметры симуляции", padding="5")
        sim_frame.pack(fill=tk.X, pady=5)
        
        # Количество дронов
        ttk.Label(sim_frame, text="Количество дронов:").pack(anchor=tk.W)
        
        # Фрейм для количества дронов и автоматического расчета
        drone_count_frame = ttk.Frame(sim_frame)
        drone_count_frame.pack(fill=tk.X, pady=5)
        
        self.drone_count_var = tk.StringVar(value="50")
        self.drone_count_entry = ttk.Entry(
            drone_count_frame,
            textvariable=self.drone_count_var,
            width=10
        )
        self.drone_count_entry.pack(side=tk.LEFT, padx=5)
        
        self.auto_count_button = ttk.Button(
            drone_count_frame,
            text="Автоматический расчет",
            command=self.auto_calculate_drones,
            state=tk.DISABLED
        )
        self.auto_count_button.pack(side=tk.LEFT, padx=5)
        
        # Кнопки управления симуляцией
        self.start_button = ttk.Button(
            self.control_frame,
            text="Запустить",
            command=self.start_simulation,
            state=tk.DISABLED
        )
        self.start_button.pack(pady=5)
        
        self.stop_button = ttk.Button(
            self.control_frame,
            text="Остановить",
            command=self.stop_simulation,
            state=tk.DISABLED
        )
        self.stop_button.pack(pady=5)
        
        # Информация о состоянии роя
        self.status_frame = ttk.LabelFrame(self.control_frame, text="Статус роя", padding="5")
        self.status_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.status_frame, text="Среднее отклонение:").pack(anchor=tk.W)
        self.error_var = tk.StringVar(value="N/A")
        ttk.Label(self.status_frame, textvariable=self.error_var).pack(anchor=tk.W)
        
        ttk.Label(self.status_frame, text="Статус сходимости:").pack(anchor=tk.W)
        self.converge_var = tk.StringVar(value="N/A")
        ttk.Label(self.status_frame, textvariable=self.converge_var).pack(anchor=tk.W)
        
        # Область визуализации
        self.visualization = Visualization3D(self.visualization_frame)
        self.visualization.set_base_coordinates(
            self.base_latitude,
            self.base_longitude,
            self.base_altitude
        )
        
    def load_image(self):
        """Загрузка изображения"""
        filename = filedialog.askopenfilename(
            filetypes=[
                ("Изображения", "*.png *.jpg *.jpeg *.bmp"),
                ("Все файлы", "*.*")
            ]
        )
        
        if filename:
            try:
                # Обработка изображения
                contour_points = self.image_processor.process_image(filename)
                
                # Активация кнопки автоматического расчета
                self.auto_count_button['state'] = tk.NORMAL
                
                # Получение рекомендуемого количества дронов
                recommended = self.image_processor.get_recommended_drone_count()
                self.drone_count_var.set(str(recommended))
                
                # Получение количества дронов из поля ввода
                drone_count = int(self.drone_count_var.get())
                
                # Равномерное распределение точек
                if len(contour_points) > drone_count:
                    # Если точек больше чем дронов, равномерно распределяем
                    evenly_distributed_points = self.image_processor.distribute_points_evenly(drone_count)
                    self.drone_swarm = DroneSwarm(drone_count, evenly_distributed_points)
                else:
                    # Иначе используем все точки контура
                    self.drone_swarm = DroneSwarm(drone_count, contour_points)
                
                # Обновление визуализации
                self.visualization.set_target_points(contour_points)
                self.visualization.set_drones(self.drone_swarm.get_drones())
                self.visualization.set_drone_swarm(self.drone_swarm)
                
                # Активация кнопок
                self.start_button['state'] = tk.NORMAL
                
                messagebox.showinfo(
                    "Успех",
                    f"Изображение загружено и обработано успешно.\n"
                    f"Выделено {len(contour_points)} точек контура.\n"
                    f"Рекомендуемое количество дронов: {recommended}"
                )
                
            except Exception as e:
                messagebox.showerror(
                    "Ошибка",
                    f"Ошибка при загрузке изображения: {str(e)}"
                )
    
    def set_base_coordinates(self):
        """Диалог установки базовых координат"""
        dialog = tk.Toplevel(self)
        dialog.title("Задать базовые координаты")
        dialog.geometry("300x200")
        dialog.transient(self)
        dialog.grab_set()
        
        # Поля ввода
        ttk.Label(dialog, text="Широта (°):").pack(pady=5)
        lat_var = tk.StringVar(value=str(self.base_latitude))
        lat_entry = ttk.Entry(dialog, textvariable=lat_var)
        lat_entry.pack()
        
        ttk.Label(dialog, text="Долгота (°):").pack(pady=5)
        lon_var = tk.StringVar(value=str(self.base_longitude))
        lon_entry = ttk.Entry(dialog, textvariable=lon_var)
        lon_entry.pack()
        
        ttk.Label(dialog, text="Высота (м):").pack(pady=5)
        alt_var = tk.StringVar(value=str(self.base_altitude))
        alt_entry = ttk.Entry(dialog, textvariable=alt_var)
        alt_entry.pack()
        
        def apply_coordinates():
            try:
                lat = float(lat_var.get())
                lon = float(lon_var.get())
                alt = float(alt_var.get())
                
                if not (-90 <= lat <= 90):
                    raise ValueError("Широта должна быть от -90° до 90°")
                if not (-180 <= lon <= 180):
                    raise ValueError("Долгота должна быть от -180° до 180°")
                
                self.base_latitude = lat
                self.base_longitude = lon
                self.base_altitude = alt
                
                self.visualization.set_base_coordinates(lat, lon, alt)
                dialog.destroy()
                
            except ValueError as e:
                messagebox.showerror("Ошибка", str(e))
        
        ttk.Button(dialog, text="Применить", command=apply_coordinates).pack(pady=10)
        
    def on_closing(self):
        """Обработка закрытия приложения"""
        if self.visualization:
            self.visualization.cleanup()
        self.quit()

if __name__ == "__main__":
    app = DroneSwarmApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()