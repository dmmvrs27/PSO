import numpy as np
from dataclasses import dataclass


@dataclass
class Drone:
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    target_position: np.ndarray
    best_fitness: float
    current_target_index: int = -1
    is_dragging: bool = False
    id: int = 0


class DroneSwarm:
    """Класс управления роем дронов"""

    def __init__(self, num_drones, target_points):
        self.num_drones = num_drones
        self.target_points = np.array(target_points)
        self.drones = []

        # Параметры PSO
        self.inertia_weight = 0.6
        self.cognitive_param = 1.2
        self.social_param = 1.4

        # Скорость движения
        self.max_velocity = 1.5
        self.slowdown_factor = 0.15

        # Личное пространство
        self.personal_space = 2.0
        self.separation_weight = 1.0

        # Создание дронов
        self.random_distribute_drones()

    # ------------------------------------------------------------

    def random_distribute_drones(self):
        self.drones = []

        if len(self.target_points) == 0:
            return

        min_vals = self.target_points.min(axis=0) - 50
        max_vals = self.target_points.max(axis=0) + 50

        for i in range(self.num_drones):
            pos = np.array([
                np.random.uniform(min_vals[0], max_vals[0]),
                np.random.uniform(min_vals[1], max_vals[1]),
                np.random.uniform(min_vals[2], max_vals[2])
            ])

            drone = Drone(
                position=pos,
                velocity=np.zeros(3),
                best_position=pos.copy(),
                target_position=self.target_points[i % len(self.target_points)],
                best_fitness=np.linalg.norm(pos - self.target_points[i % len(self.target_points)]),
                id=i
            )
            self.drones.append(drone)

    # ------------------------------------------------------------

    def assign_targets_dynamically(self):
        """Каждый дрон выбирает ближайшую свободную цель"""

        if len(self.target_points) == 0:
            return

        occupied = set()

        # сначала сбрасываем текущие цели
        for drone in self.drones:
            drone.current_target_index = -1

        for drone in self.drones:
            min_dist = float("inf")
            best_idx = -1

            for idx, target in enumerate(self.target_points):
                if idx in occupied:
                    continue
                dist = np.linalg.norm(drone.position - target)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx

            if best_idx != -1:
                drone.current_target_index = best_idx
                drone.target_position = self.target_points[best_idx]
                occupied.add(best_idx)

    # ------------------------------------------------------------

    def compute_separation(self, drone):
        """Сила отталкивания от соседних дронов"""
        separation = np.zeros(3)

        for other in self.drones:
            if other is drone:
                continue

            diff = drone.position - other.position
            dist = np.linalg.norm(diff)

            if 0 < dist < self.personal_space:
                separation += (diff / dist) * (self.personal_space - dist)

        return separation * self.separation_weight

    # ------------------------------------------------------------

    def update_positions(self):
        """Основной шаг симуляции"""

        # Назначаем ближайшие свободные цели
        self.assign_targets_dynamically()

        for drone in self.drones:
            if drone.is_dragging:
                continue

            r1 = np.random.random(3)
            r2 = np.random.random(3)

            cognitive = self.cognitive_param * r1 * (drone.best_position - drone.position)
            social = self.social_param * r2 * (drone.target_position - drone.position)
            separation = self.compute_separation(drone)

            drone.velocity = (
                self.inertia_weight * drone.velocity +
                cognitive +
                social +
                separation
            )

            # Ограничение скорости
            speed = np.linalg.norm(drone.velocity)
            if speed > self.max_velocity:
                drone.velocity = drone.velocity / speed * self.max_velocity

            # Перемещение
            drone.position += drone.velocity * self.slowdown_factor

            # Обновление личного лучшего
            fitness = np.linalg.norm(drone.position - drone.target_position)
            if fitness < drone.best_fitness:
                drone.best_fitness = fitness
                drone.best_position = drone.position.copy()

    # ------------------------------------------------------------

    def get_drones(self):
        """Для визуализации"""
        return [(drone.position, drone.target_position, drone.id) for drone in self.drones]

    # ------------------------------------------------------------

    def get_average_error(self):
        errors = [np.linalg.norm(d.position - d.target_position) for d in self.drones]
        return np.mean(errors)

    def is_converged(self, threshold=1.0):
        return self.get_average_error() < threshold

    # ------------------------------------------------------------
    # функции для перетаскивания мышью

    def set_drone_position(self, drone_index, new_position):
        if 0 <= drone_index < len(self.drones):
            self.drones[drone_index].position = new_position

    def start_dragging(self, drone_index):
        if 0 <= drone_index < len(self.drones):
            self.drones[drone_index].is_dragging = True
            self.drones[drone_index].velocity = np.zeros(3)

    def stop_dragging(self, drone_index):
        if 0 <= drone_index < len(self.drones):
            self.drones[drone_index].is_dragging = False

    # ------------------------------------------------------------

    def get_drone_info(self, drone_index):
        if 0 <= drone_index < len(self.drones):
            d = self.drones[drone_index]
            return {
                "position": d.position,
                "velocity": d.velocity,
                "target": d.target_position,
                "fitness": d.best_fitness,
                "id": d.id
            }
        return None