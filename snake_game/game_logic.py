import random
import math
import heapq
from enum import Enum
from typing import List, Tuple, Optional
import time

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class GamePoint:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __repr__(self):
        return f"({self.x}, {self.y})"

class Node:
    def __init__(self, pos: GamePoint, g: int, h: int, parent=None):
        self.pos = pos
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
    
    def __lt__(self, other):
        return self.f < other.f

class SnakeGameLogic:
    def __init__(self):
        # Игровые константы
        self.CELL_SIZE = 20
        self.GRID_WIDTH = 30
        self.GRID_HEIGHT = 20
        self.MAX_SCORE = 200
        
        # Игровые переменные
        self.grid = []
        self.snake = []
        self.food = GamePoint()
        self.special_food = GamePoint()
        self.special_food_active = False
        self.special_food_type = 1
        self.score = 0
        self.direction = Direction.RIGHT
        self.next_direction = Direction.RIGHT
        self.game_running = False
        self.game_paused = False
        self.game_over = False
        self.won = False
        
        # Улучшения
        self.power_up_duration = 0
        self.power_up_active = False
        self.ghost_mode = False
        self.slow_mode = False
        self.speed_mode = False
        self.hard_mode = False
        self.auto_play = False
        self.use_aco = False
        
        # Параметры ACO
        self.pheromone_matrix = []
        self.alpha = 1.0
        self.beta = 2.0
        self.evaporation_rate = 0.5
        self.num_ants = 10
        self.max_iterations = 50
        
        # Скорости
        self.game_speed = 150  # мс
        self.special_food_chance = 200
        
        # Для отслеживания улучшений
        self.current_power_up_text = ""
        self.power_up_end_time = 0
        
        # Инициализация
        self.initialize_grid()
    
    def initialize_grid(self):
        """Инициализация игровой сетки"""
        if self.hard_mode:
            self.GRID_WIDTH = 15
            self.GRID_HEIGHT = 10
            self.game_speed = 90
            self.MAX_SCORE = 300
            self.special_food_chance = 70
        else:
            self.GRID_WIDTH = 30
            self.GRID_HEIGHT = 20
            self.game_speed = 150
            self.MAX_SCORE = 200
            self.special_food_chance = 200
        
        self.grid = [[0 for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        
        # Инициализация матрицы феромонов для ACO
        self.pheromone_matrix = [[0.1 for _ in range(self.GRID_WIDTH)] 
                                for _ in range(self.GRID_HEIGHT)]
    
    def reset_game(self):
        """Сброс игры в начальное состояние"""
        self.snake.clear()
        
        # Начальная позиция змейки
        start_x = self.GRID_WIDTH // 2
        start_y = self.GRID_HEIGHT // 2
        start_x = max(3, min(start_x, self.GRID_WIDTH - 1))
        start_y = max(0, min(start_y, self.GRID_HEIGHT - 1))
        
        # Создаем змейку из 4 сегментов
        self.snake.append(GamePoint(start_x, start_y))
        self.snake.append(GamePoint(start_x - 1, start_y))
        self.snake.append(GamePoint(start_x - 2, start_y))
        self.snake.append(GamePoint(start_x - 3, start_y))
        
        self.direction = Direction.RIGHT
        self.next_direction = Direction.RIGHT
        self.score = 0
        self.game_running = False
        self.game_paused = False
        self.game_over = False
        self.won = False
        
        # Сброс улучшений
        self.power_up_active = False
        self.ghost_mode = False
        self.slow_mode = False
        self.speed_mode = False
        self.power_up_duration = 0
        self.special_food_active = False
        self.current_power_up_text = ""
        self.power_up_end_time = 0
        
        # Настройка скорости в зависимости от режима
        self.game_speed = 90 if self.hard_mode else 150
        self.MAX_SCORE = 300 if self.hard_mode else 200
        self.special_food_chance = 70 if self.hard_mode else 200
        
        self.initialize_grid()
        self.generate_food()
    
    def generate_food(self):
        """Генерация обычной еды"""
        available_cells = []
        
        # Собираем все доступные клетки
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                occupied = False
                
                # Проверяем, не занята ли клетка змейкой
                for segment in self.snake:
                    if segment.x == x and segment.y == y:
                        occupied = True
                        break
                
                # Если клетка свободна и не препятствие
                if not occupied and self.grid[y][x] != 1:
                    available_cells.append(GamePoint(x, y))
        
        if available_cells:
            # Выбираем случайную доступную клетку
            self.food = random.choice(available_cells)
        else:
            # Экстренная генерация
            for y in range(self.GRID_HEIGHT):
                for x in range(self.GRID_WIDTH):
                    occupied = False
                    for segment in self.snake:
                        if segment.x == x and segment.y == y:
                            occupied = True
                            break
                    if not occupied:
                        self.food = GamePoint(x, y)
                        return
            
            # Если вообще нет свободных клеток
            if self.snake:
                self.food = self.snake[-1]
    
    def generate_special_food(self):
        """Генерация специального фрукта"""
        if not self.game_running or self.game_paused:
            return False
        
        # Проверяем вероятность появления (1/шанс)
        if not self.special_food_active and random.randint(1, self.special_food_chance) == 1:
            available_cells = []
            
            # Ищем доступные клетки
            for y in range(self.GRID_HEIGHT):
                for x in range(self.GRID_WIDTH):
                    occupied = False
                    
                    # Проверяем змейку
                    for segment in self.snake:
                        if segment.x == x and segment.y == y:
                            occupied = True
                            break
                    
                    # Проверяем обычную еду
                    if self.food.x == x and self.food.y == y:
                        occupied = True
                    
                    # Если клетка свободна и не препятствие
                    if not occupied and self.grid[y][x] != 1:
                        available_cells.append(GamePoint(x, y))
            
            if available_cells:
                # Выбираем случайную клетку
                self.special_food = random.choice(available_cells)
                self.special_food_active = True
                
                # Определяем тип спецфрукта
                if self.hard_mode:
                    # Случайный тип в сложном режиме
                    self.special_food_type = random.randint(1, 3)
                else:
                    # Взвешенная вероятность в обычном режиме
                    type_chance = random.randint(1, 100)
                    if type_chance <= 40:
                        self.special_food_type = 1  # Ускорение
                    elif type_chance <= 70:
                        self.special_food_type = 2  # Призрак
                    else:
                        self.special_food_type = 3  # Замедление
                
                return True  # Спецфрукт был создан
        return False  # Спецфрукт не был создан
    
    def get_food_score(self):
        """Получить очки за обычную еду"""
        if not self.hard_mode:
            return 10
        else:
            if self.score < 100:
                return 10
            elif self.score < 200:
                return 15
            else:
                return 20
    
    def get_special_food_score(self):
        """Получить очки за спецфрукт"""
        if not self.hard_mode:
            return 5
        else:
            if self.score < 100:
                return 5
            elif self.score < 200:
                return 10
            else:
                return 15
    
    def get_food_segments(self):
        """Получить сегменты за обычную еду"""
        if not self.hard_mode:
            return 3
        else:
            if self.score < 100:
                return 2
            elif self.score < 200:
                return 3
            else:
                return 5
    
    def get_special_food_segments(self):
        """Получить сегменты за спецфрукт"""
        return 1
    
    def move_snake(self):
        """Движение змейки"""
        if self.game_over or self.game_paused or not self.game_running:
            return False
        
        # Обновляем направление
        self.direction = self.next_direction
        
        if not self.snake:
            return False
        
        head = self.snake[0]
        new_head = GamePoint(head.x, head.y)
        
        # Вычисляем новую позицию головы
        if self.direction == Direction.UP:
            new_head.y -= 1
        elif self.direction == Direction.RIGHT:
            new_head.x += 1
        elif self.direction == Direction.DOWN:
            new_head.y += 1
        elif self.direction == Direction.LEFT:
            new_head.x -= 1
        
        # Телепортация в обычном режиме ИЛИ в режиме призрака (даже в сложном режиме)
        if not self.hard_mode or self.ghost_mode:  # Изменено условие
            if new_head.x < 0:
                new_head.x = self.GRID_WIDTH - 1
            elif new_head.x >= self.GRID_WIDTH:
                new_head.x = 0
            if new_head.y < 0:
                new_head.y = self.GRID_HEIGHT - 1
            elif new_head.y >= self.GRID_HEIGHT:
                new_head.y = 0
        
        # Проверка границ в сложном режиме без призрака
        if self.hard_mode and not self.ghost_mode:
            if (new_head.x < 0 or new_head.x >= self.GRID_WIDTH or
                new_head.y < 0 or new_head.y >= self.GRID_HEIGHT):
                self.game_over = True
                return False
        
        # Добавляем новую голову
        self.snake.insert(0, new_head)
        
        # Проверяем съели ли что-то
        ate_food = (new_head.x == self.food.x and new_head.y == self.food.y)
        ate_special_food = (self.special_food_active and 
                        new_head.x == self.special_food.x and 
                        new_head.y == self.special_food.y)
        
        if not ate_food and not ate_special_food:
            # Удаляем хвост если ничего не съели
            self.snake.pop()
        else:
            if ate_food:
                # Съели обычную еду
                self.score += self.get_food_score()
                segments_to_add = self.get_food_segments()
                
                # Добавляем сегменты
                for _ in range(segments_to_add - 1):
                    self.snake.append(GamePoint(self.snake[-1].x, self.snake[-1].y))
                
                self.generate_food()
                self.ensure_food_exists()
            
            if ate_special_food:
                # Съели спецфрукт
                self.score += self.get_special_food_score()
                self.special_food_active = False
                
                # Добавляем сегменты
                segments_to_add = self.get_special_food_segments()
                for _ in range(segments_to_add):
                    self.snake.append(GamePoint(self.snake[-1].x, self.snake[-1].y))
                
                # Применяем эффект
                power_up_applied = False
                if self.hard_mode:
                    # Сложный режим
                    if self.special_food_type == 1:  # Ускорение
                        self.speed_mode = True
                        self.game_speed = 24
                        power_up_applied = True
                        self.power_up_duration = 2000 // self.game_speed
                        self.current_power_up_text = "УСКОРЕНИЕ!"
                    elif self.special_food_type == 2:  # Призрак
                        self.ghost_mode = True
                        power_up_applied = True
                        self.power_up_duration = 4000 // self.game_speed
                        self.current_power_up_text = "РЕЖИМ ПРИЗРАКА!"
                    else:  # Замедление
                        self.slow_mode = True
                        self.game_speed = 120
                        power_up_applied = True
                        self.power_up_duration = 3000 // self.game_speed
                        self.current_power_up_text = "ЗАМЕДЛЕНИЕ!"
                else:
                    # Обычный режим
                    if self.special_food_type == 1:  # Ускорение
                        self.speed_mode = True
                        self.game_speed = 75
                        power_up_applied = True
                        self.power_up_duration = 6000 // self.game_speed
                        self.current_power_up_text = "УСКОРЕНИЕ!"
                    elif self.special_food_type == 2:  # Призрак
                        self.ghost_mode = True
                        power_up_applied = True
                        self.power_up_duration = 6000 // self.game_speed
                        self.current_power_up_text = "РЕЖИМ ПРИЗРАКА!"
                    else:  # Замедление
                        self.slow_mode = True
                        self.game_speed = 300
                        power_up_applied = True
                        self.power_up_duration = 6000 // self.game_speed
                        self.current_power_up_text = "ЗАМЕДЛЕНИЕ!"
                
                if power_up_applied:
                    self.power_up_active = True
                    # Устанавливаем время окончания улучшения
                    self.power_up_end_time = time.time() * 1000 + (self.power_up_duration * self.game_speed)
                    return True  # Возвращаем True, если было применено улучшение
        
        # Проверяем столкновения
        self.check_collision()
        
        # Проверяем победу
        if self.score >= self.MAX_SCORE and not self.won:
            self.won = True
            self.game_running = False
    
        return False  # Возвращаем False, если улучшение не было применено
    def check_collision(self):
        """Проверка столкновений"""
        if not self.snake:
            return
        
        head = self.snake[0]
        
        # Проверка столкновения с телом
        for i in range(1, len(self.snake)):
            if head.x == self.snake[i].x and head.y == self.snake[i].y:
                if not self.hard_mode and self.ghost_mode:
                    continue
                self.game_over = True
                return
        
        # Проверка столкновения с границами (только если не в режиме призрака)
        if not self.ghost_mode:  # Изменено условие
            if (head.x < 0 or head.x >= self.GRID_WIDTH or
                head.y < 0 or head.y >= self.GRID_HEIGHT):
                self.game_over = True
                return
    def is_cell_walkable(self, x: int, y: int) -> bool:
        """Проверка проходимости клетки"""
        # В режиме призрака всегда можно ходить сквозь стены
        if self.ghost_mode:
            return True
        
        # Проверка границ
        if x < 0 or x >= self.GRID_WIDTH or y < 0 or y >= self.GRID_HEIGHT:
            return False
        
        # Проверка тела змейки
        cell = GamePoint(x, y)
        for i in range(len(self.snake) - 1):
            if self.snake[i] == cell:
                return False
        
        return True
    def find_path_with_astar(self, start: GamePoint, end: GamePoint) -> List[GamePoint]:
        """Поиск пути с использованием алгоритма A*"""
        # Эвристическая функция
        def heuristic(a: GamePoint, b: GamePoint) -> int:
            # В режиме призрака учитываем телепортацию даже в сложном режиме
            if not self.hard_mode or self.ghost_mode:  # Изменено условие
                # Учет телепортации
                dx = abs(a.x - b.x)
                dy = abs(a.y - b.y)
                dx = min(dx, self.GRID_WIDTH - dx)
                dy = min(dy, self.GRID_HEIGHT - dy)
                return dx + dy
            return abs(a.x - b.x) + abs(a.y - b.y)
        
        # Структуры данных
        open_list = []
        closed_set = set()
        node_map = {}
        
        # Начальный узел
        start_node = Node(start, 0, heuristic(start, end), None)
        heapq.heappush(open_list, (start_node.f, start_node))
        node_map[(start.x, start.y)] = start_node
        
        # Возможные направления
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        while open_list:
            _, current = heapq.heappop(open_list)
            
            # Проверка достижения цели
            if current.pos == end:
                # Восстановление пути
                path = []
                node = current
                while node:
                    path.append(node.pos)
                    node = node.parent
                return list(reversed(path))
            
            if (current.pos.x, current.pos.y) in closed_set:
                continue
            
            closed_set.add((current.pos.x, current.pos.y))
            
            # Обработка соседей
            for dx, dy in directions:
                neighbor_pos = GamePoint(current.pos.x + dx, current.pos.y + dy)
                
                # Телепортация в обычном режиме ИЛИ в режиме призрака
                if not self.hard_mode or self.ghost_mode:  # Изменено условие
                    if neighbor_pos.x < 0:
                        neighbor_pos.x = self.GRID_WIDTH - 1
                    elif neighbor_pos.x >= self.GRID_WIDTH:
                        neighbor_pos.x = 0
                    if neighbor_pos.y < 0:
                        neighbor_pos.y = self.GRID_HEIGHT - 1
                    elif neighbor_pos.y >= self.GRID_HEIGHT:
                        neighbor_pos.y = 0
                
                # Проверка границ (только если не в режиме призрака)
                if not self.ghost_mode:  # Изменено условие
                    if (neighbor_pos.x < 0 or neighbor_pos.x >= self.GRID_WIDTH or
                        neighbor_pos.y < 0 or neighbor_pos.y >= self.GRID_HEIGHT):
                        continue
                
                # Проверка проходимости
                if not self.is_cell_walkable(neighbor_pos.x, neighbor_pos.y):
                    continue
                
                if (neighbor_pos.x, neighbor_pos.y) in closed_set:
                    continue
                
                # Вычисление стоимости
                new_g = current.g + 1
                
                if (neighbor_pos.x, neighbor_pos.y) not in node_map:
                    # Создание нового узла
                    neighbor_node = Node(
                        neighbor_pos,
                        new_g,
                        heuristic(neighbor_pos, end),
                        current
                    )
                    heapq.heappush(open_list, (neighbor_node.f, neighbor_node))
                    node_map[(neighbor_pos.x, neighbor_pos.y)] = neighbor_node
                else:
                    # Обновление существующего узла
                    existing_node = node_map[(neighbor_pos.x, neighbor_pos.y)]
                    if new_g < existing_node.g:
                        existing_node.g = new_g
                        existing_node.f = new_g + existing_node.h
                        existing_node.parent = current
        
        return []  # Путь не найден
    def find_path_with_aco(self, start: GamePoint, end: GamePoint) -> List[GamePoint]:
        """Поиск пути с использованием алгоритма муравьиной колонии"""
        if start == end:
            return [start]
        
        if not self.pheromone_matrix:
            self.pheromone_matrix = [[0.1 for _ in range(self.GRID_WIDTH)] 
                                    for _ in range(self.GRID_HEIGHT)]
        
        best_path = []
        best_path_length = float('inf')
        
        for _ in range(self.max_iterations):
            trial_pheromones = [row[:] for row in self.pheromone_matrix]
            visited = [[False for _ in range(self.GRID_WIDTH)] 
                    for _ in range(self.GRID_HEIGHT)]
            
            current_path = []
            current_pos = start
            current_path.append(current_pos)
            visited[current_pos.y][current_pos.x] = True
            
            while current_pos != end:
                neighbors = []
                probabilities = []
                total_prob = 0.0
                
                directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
                
                for dx, dy in directions:
                    next_pos = GamePoint(current_pos.x + dx, current_pos.y + dy)
                    
                    # Телепортация в обычном режиме ИЛИ в режиме призрака
                    if not self.hard_mode or self.ghost_mode:  # Изменено условие
                        if next_pos.x < 0:
                            next_pos.x = self.GRID_WIDTH - 1
                        elif next_pos.x >= self.GRID_WIDTH:
                            next_pos.x = 0
                        if next_pos.y < 0:
                            next_pos.y = self.GRID_HEIGHT - 1
                        elif next_pos.y >= self.GRID_HEIGHT:
                            next_pos.y = 0
                    
                    # Проверка проходимости и посещения
                    # В режиме призрака не проверяем границы
                    if (self.ghost_mode or  # Изменено условие
                        (0 <= next_pos.x < self.GRID_WIDTH and 
                        0 <= next_pos.y < self.GRID_HEIGHT)):
                        
                        if (self.is_cell_walkable(next_pos.x, next_pos.y) and
                            not visited[next_pos.y][next_pos.x]):
                            
                            neighbors.append(next_pos)
                            # Эвристика (обратное расстояние до цели)
                            # В режиме призрака учитываем телепортацию
                            if not self.hard_mode or self.ghost_mode:  # Изменено условие
                                dx_dist = abs(next_pos.x - end.x)
                                dy_dist = abs(next_pos.y - end.y)
                                dx_dist = min(dx_dist, self.GRID_WIDTH - dx_dist)
                                dy_dist = min(dy_dist, self.GRID_HEIGHT - dy_dist)
                                heuristic = 1.0 / (dx_dist + dy_dist + 1)
                            else:
                                heuristic = 1.0 / (abs(next_pos.x - end.x) + 
                                                abs(next_pos.y - end.y) + 1)
                            
                            pheromone = trial_pheromones[next_pos.y][next_pos.x]
                            prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
                            probabilities.append(prob)
                            total_prob += prob
                
                if not neighbors:
                    break
                
                # Нормализация вероятностей
                if total_prob > 0:
                    probabilities = [p / total_prob for p in probabilities]
                
                # Вероятностный выбор
                rand_val = random.random()
                cumulative_prob = 0.0
                chosen_idx = 0
                
                for i, prob in enumerate(probabilities):
                    cumulative_prob += prob
                    if rand_val <= cumulative_prob:
                        chosen_idx = i
                        break
                
                current_pos = neighbors[chosen_idx]
                current_path.append(current_pos)
                visited[current_pos.y][current_pos.x] = True
                trial_pheromones[current_pos.y][current_pos.x] += 1.0
            
            # Оценка пути
            if current_pos == end:
                path_length = len(current_path)
                if path_length < best_path_length:
                    best_path = current_path
                    best_path_length = path_length
        
        # Глобальное обновление феромонов
        if best_path:
            # Испарение феромонов
            for y in range(self.GRID_HEIGHT):
                for x in range(self.GRID_WIDTH):
                    self.pheromone_matrix[y][x] *= (1.0 - self.evaporation_rate)
            
            # Добавление феромонов вдоль лучшего пути
            delta_pheromone = 100.0 / best_path_length
            for point in best_path:
                if (0 <= point.x < self.GRID_WIDTH and 
                    0 <= point.y < self.GRID_HEIGHT):
                    self.pheromone_matrix[point.y][point.x] += delta_pheromone
        
        return best_path
    def get_next_direction(self) -> GamePoint:
        """Получение следующего направления для автопилота"""
        if not self.snake:
            return GamePoint(0, 0)
        
        head = self.snake[0]
        target = self.food
        
        # Выбор цели
        if self.special_food_active:
            if not self.hard_mode:
                # Избегаем замедление в обычном режиме
                if self.special_food_type != 3:
                    target = self.special_food
            else:
                # 50% шанс взять спецфрукт в сложном режиме
                if random.random() < 0.5:
                    target = self.special_food
        
        # Поиск пути
        path = []
        if self.use_aco:
            path = self.find_path_with_aco(head, target)
        else:
            path = self.find_path_with_astar(head, target)
        
        # Определение направления
        if len(path) > 1:
            next_step = path[1]
            if next_step.x > head.x:
                return GamePoint(1, 0)  # Вправо
            elif next_step.x < head.x:
                return GamePoint(-1, 0)  # Влево
            elif next_step.y > head.y:
                return GamePoint(0, 1)  # Вниз
            elif next_step.y < head.y:
                return GamePoint(0, -1)  # Вверх
        
        # Резервная логика
        if self.direction == Direction.UP:
            return GamePoint(0, -1)
        elif self.direction == Direction.RIGHT:
            return GamePoint(1, 0)
        elif self.direction == Direction.DOWN:
            return GamePoint(0, 1)
        elif self.direction == Direction.LEFT:
            return GamePoint(-1, 0)
        
        return GamePoint(1, 0)
    
    def update_auto_play_direction(self):
        """Обновление направления для автопилота"""
        if not self.auto_play or not self.game_running or self.game_paused or self.game_over:
            return
        
        next_dir = self.get_next_direction()
        
        # Преобразование вектора в направление
        if next_dir.x == 1 and next_dir.y == 0:  # Вправо
            if self.direction != Direction.LEFT:
                self.next_direction = Direction.RIGHT
        elif next_dir.x == -1 and next_dir.y == 0:  # Влево
            if self.direction != Direction.RIGHT:
                self.next_direction = Direction.LEFT
        elif next_dir.x == 0 and next_dir.y == 1:  # Вниз
            if self.direction != Direction.UP:
                self.next_direction = Direction.DOWN
        elif next_dir.x == 0 and next_dir.y == -1:  # Вверх
            if self.direction != Direction.DOWN:
                self.next_direction = Direction.UP
    
    def ensure_food_exists(self):
        """Гарантия существования еды"""
        food_valid = False
        
        # Проверяем, что еда в пределах поля
        if (0 <= self.food.x < self.GRID_WIDTH and 
            0 <= self.food.y < self.GRID_HEIGHT):
            food_valid = True
            
            # Проверяем, что еда не на змейке
            for segment in self.snake:
                if segment == self.food:
                    food_valid = False
                    break
        
        if not food_valid:
            self.generate_food()
    
    def update_power_ups(self):
        """Обновление состояния улучшений"""
        if self.power_up_active:
            # Проверяем время
            current_time = time.time() * 1000
            if current_time >= self.power_up_end_time:
                self.power_up_active = False
                self.ghost_mode = False
                self.slow_mode = False
                self.speed_mode = False
                self.game_speed = 90 if self.hard_mode else 150
                self.current_power_up_text = ""
                return True
        return False
    
    def get_power_up_text(self):
        """Получить текст текущего улучшения"""
        return self.current_power_up_text
    
    def is_power_up_active(self):
        """Проверка, активно ли улучшение"""
        return self.power_up_active and self.current_power_up_text != ""