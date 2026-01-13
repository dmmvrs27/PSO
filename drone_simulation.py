import numpy as np
from dataclasses import dataclass

@dataclass
class Drone:
    """Класс для представления отдельного дрона"""
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    target_position: np.ndarray
    best_fitness: float
    is_dragging: bool = False  # Поле для отслеживания перетаскивания
    id: int = 0  # Уникальный идентификатор дрона

class DroneSwarm:
    """Класс для управления роем дронов"""
    def __init__(self, num_drones, target_points):
        self.num_drones = num_drones
        self.target_points = np.array(target_points)
        self.drones = []
        
        # Параметры роевого алгоритма
        self.inertia_weight = 0.5
        self.cognitive_param = 0.3
        self.social_param = 0.3
        
        # Параметр для определения, какой дрон выбран
        self.selected_drone = None
        
        # Случайное начальное размещение дронов
        self.random_distribute_drones()
        
    def random_distribute_drones(self):
        """Случайное начальное размещение дронов перед стягиванием в фигуру"""
        self.drones = []
        
        # Определяем границы пространства для случайного размещения
        # Находим ограничивающий прямоугольник для целевых точек
        min_x = min([point[0] for point in self.target_points]) if len(self.target_points) > 0 else -50
        max_x = max([point[0] for point in self.target_points]) if len(self.target_points) > 0 else 50
        min_y = min([point[1] for point in self.target_points]) if len(self.target_points) > 0 else 0
        max_y = max([point[1] for point in self.target_points]) if len(self.target_points) > 0 else 30
        min_z = min([point[2] for point in self.target_points]) if len(self.target_points) > 0 else -50
        max_z = max([point[2] for point in self.target_points]) if len(self.target_points) > 0 else 50
        
        # Расширяем границы для более видимого эффекта стягивания
        boundary_extension = 50
        min_x -= boundary_extension
        max_x += boundary_extension
        min_y -= boundary_extension/2  # Меньше расширяем по Y, чтобы дроны не уходили глубоко под землю
        max_y += boundary_extension
        min_z -= boundary_extension
        max_z += boundary_extension
        
        # Распределяем целевые точки между дронами
        if self.num_drones <= len(self.target_points):
            # Если дронов меньше или равно количеству точек, распределяем равномерно
            step = len(self.target_points) / self.num_drones
            for i in range(self.num_drones):
                idx = int(i * step)
                target = self.target_points[idx]
                # Случайная начальная позиция в расширенных границах
                random_pos = np.array([
                    np.random.uniform(min_x, max_x),
                    np.random.uniform(min_y, max_y),
                    np.random.uniform(min_z, max_z)
                ])
                
                drone = Drone(
                    position=random_pos,
                    velocity=np.zeros(3),
                    best_position=random_pos.copy(),
                    target_position=target,
                    best_fitness=np.linalg.norm(random_pos - target),
                    id=i
                )
                self.drones.append(drone)
        else:
            # Если дронов больше, чем точек, сначала распределяем по одному дрону на точку
            for i, target in enumerate(self.target_points):
                # Случайная начальная позиция
                random_pos = np.array([
                    np.random.uniform(min_x, max_x),
                    np.random.uniform(min_y, max_y),
                    np.random.uniform(min_z, max_z)
                ])
                
                drone = Drone(
                    position=random_pos,
                    velocity=np.zeros(3),
                    best_position=random_pos.copy(),
                    target_position=target,
                    best_fitness=np.linalg.norm(random_pos - target),
                    id=i
                )
                self.drones.append(drone)
            
            # Оставшиеся дроны распределяем случайно по существующим целям
            for i in range(len(self.target_points), self.num_drones):
                target_idx = i % len(self.target_points)
                target = self.target_points[target_idx]
                # Случайная начальная позиция
                random_pos = np.array([
                    np.random.uniform(min_x, max_x),
                    np.random.uniform(min_y, max_y),
                    np.random.uniform(min_z, max_z)
                ])
                
                drone = Drone(
                    position=random_pos,
                    velocity=np.zeros(3),
                    best_position=random_pos.copy(),
                    target_position=target,
                    best_fitness=np.linalg.norm(random_pos - target),
                    id=i
                )
                self.drones.append(drone)
    
    def update_positions(self):
        """Обновление позиций всех дронов"""
        for drone in self.drones:
            # Пропускаем обновление, если дрон перетаскивается
            if drone.is_dragging:
                continue
                
            # Случайные коэффициенты
            r1 = np.random.random(3)
            r2 = np.random.random(3)
            
            # Обновление скорости
            cognitive_velocity = self.cognitive_param * r1 * (
                drone.best_position - drone.position
            )
            social_velocity = self.social_param * r2 * (
                drone.target_position - drone.position
            )
            
            drone.velocity = (self.inertia_weight * drone.velocity +
                            cognitive_velocity + social_velocity)
            
            # Ограничение максимальной скорости
            max_velocity = 2.0
            velocity_magnitude = np.linalg.norm(drone.velocity)
            if velocity_magnitude > max_velocity:
                drone.velocity = (drone.velocity / velocity_magnitude) * max_velocity
            
            # Обновление позиции
            slowdown_factor = 0.2
            drone.position += drone.velocity * slowdown_factor

            # Вычисление фитнес-функции (расстояние до цели)
            current_fitness = np.linalg.norm(
                drone.position - drone.target_position
            )
            
            # Обновление лучшей позиции
            if current_fitness < drone.best_fitness:
                drone.best_fitness = current_fitness
                drone.best_position = drone.position.copy()
    
    def reassign_targets(self):
        """Перераспределение целевых точек между дронами"""
        # Матрица расстояний между всеми дронами и целевыми точками
        distances = np.zeros((len(self.drones), len(self.target_points)))
        
        for i, drone in enumerate(self.drones):
            for j, target in enumerate(self.target_points):
                distances[i, j] = np.linalg.norm(drone.position - target)
        
        # Простой жадный алгоритм назначения
        assigned_targets = set()
        for _ in range(min(len(self.drones), len(self.target_points))):
            # Находим минимальное расстояние
            min_val = float('inf')
            min_i, min_j = -1, -1
            
            for i in range(len(self.drones)):
                for j in range(len(self.target_points)):
                    if j not in assigned_targets and distances[i, j] < min_val:
                        min_val = distances[i, j]
                        min_i, min_j = i, j
            
            if min_i != -1 and min_j != -1:
                self.drones[min_i].target_position = self.target_points[min_j]
                assigned_targets.add(min_j)
                # Устанавливаем большое значение, чтобы этот дрон больше не выбирался
                distances[min_i, :] = float('inf')
    
    def get_drones(self):
        """Получение текущих позиций всех дронов"""
        return [(drone.position, drone.target_position, drone.id) for drone in self.drones]
    
    def get_average_error(self):
        """Получение среднего отклонения от целевых позиций"""
        errors = [np.linalg.norm(drone.position - drone.target_position)
                 for drone in self.drones]
        return np.mean(errors)
    
    def is_converged(self, threshold=1.0):
        """Проверка сходимости роя"""
        return self.get_average_error() < threshold
        
    def set_drone_position(self, drone_index, new_position):
        """Установка новой позиции дрона при перетаскивании"""
        if 0 <= drone_index < len(self.drones):
            self.drones[drone_index].position = new_position
            
    def start_dragging(self, drone_index):
        """Начало перетаскивания дрона"""
        if 0 <= drone_index < len(self.drones):
            self.drones[drone_index].is_dragging = True
            self.drones[drone_index].velocity = np.zeros(3)
            
    def stop_dragging(self, drone_index):
        """Окончание перетаскивания дрона"""
        if 0 <= drone_index < len(self.drones):
            self.drones[drone_index].is_dragging = False
            
    def get_drone_info(self, drone_index):
        """Получение полной информации о дроне"""
        if 0 <= drone_index < len(self.drones):
            drone = self.drones[drone_index]
            return {
                'position': drone.position,
                'velocity': drone.velocity,
                'target': drone.target_position,
                'fitness': drone.best_fitness,
                'id': drone.id
            }
        return None