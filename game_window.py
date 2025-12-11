import sys
import random
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, 
                             QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QRadioButton, QGroupBox, 
                             QCheckBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QFontMetrics
from PyQt5 import uic
from game_logic import SnakeGameLogic, Direction, GamePoint

class SnakeGameWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Загружаем интерфейс из файла
        uic.loadUi('ui_snake_game.ui', self)
        
        # Инициализируем игровую логику
        self.game = SnakeGameLogic()
        
        # Настраиваем таймеры
        self.game_timer = QTimer()
        self.game_timer.timeout.connect(self.game_tick)
        
        self.special_food_timer = QTimer()
        self.special_food_timer.timeout.connect(self.generate_special_food_tick)
        self.special_food_timer.start(100)  # Каждые 100 мс
        
        self.power_up_text_timer = QTimer()
        self.power_up_text_timer.timeout.connect(self.update_power_up_text)
        
        # Переменные для текста улучшений (как в .cpp)
        self.power_up_text = ""
        self.power_up_text_position = 25
        self.power_up_color = QColor(255, 255, 255)  # Белый по умолчанию
        
        # Подключаем обработчики событий
        self.connect_signals()
        
        # Настраиваем начальное состояние
        self.update_ui()
        
        # Устанавливаем фокус для обработки клавиш
        self.setFocusPolicy(Qt.StrongFocus)
    
    def connect_signals(self):
        """Подключение обработчиков событий"""
        # Кнопки
        self.startButton.clicked.connect(self.start_game)
        self.pauseButton.clicked.connect(self.pause_game)
        self.restartButton.clicked.connect(self.restart_game)
        
        # Радиокнопки режима
        self.normalModeRadio.toggled.connect(self.on_mode_changed)
        self.hardModeRadio.toggled.connect(self.on_mode_changed)
        
        # Радиокнопки алгоритма
        self.astarRadio.toggled.connect(self.on_algorithm_changed)
        self.acoRadio.toggled.connect(self.on_algorithm_changed)
        
        # Чекбокс автопилота
        self.autoPlayCheckBox.stateChanged.connect(self.on_autoplay_changed)
    
    def start_game(self):
        """Начать игру"""
        self.game.reset_game()
        self.game.game_running = True
        self.game.game_paused = False
        
        self.startButton.setEnabled(False)
        self.pauseButton.setEnabled(True)
        self.restartButton.setEnabled(True)
        
        # Запускаем таймер игры
        self.game_timer.start(self.game.game_speed)
        
        # Скрываем сообщения
        self.gameOverLabel.setVisible(False)
        self.winLabel.setVisible(False)
        
        # Очищаем текст улучшений
        self.powerUpLabel.setText("")
        self.power_up_text = ""
        
        self.update_ui()
    
    def pause_game(self):
        """Пауза/продолжение игры"""
        if self.game.game_running:
            self.game.game_paused = not self.game.game_paused
            
            if self.game.game_paused:
                self.game_timer.stop()
                self.pauseButton.setText("Продолжить")
            else:
                self.game_timer.start(self.game.game_speed)
                self.pauseButton.setText("Пауза")
    
    def restart_game(self):
        """Перезапуск игры"""
        self.game.reset_game()
        self.game_timer.stop()
        
        self.startButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        self.pauseButton.setText("Пауза")
        
        # Сбрасываем состояние игры
        self.game.game_running = False
        self.game.game_paused = False
        
        self.gameOverLabel.setVisible(False)
        self.winLabel.setVisible(False)
        
        # Очищаем текст улучшений
        self.powerUpLabel.setText("")
        self.power_up_text = ""
        if self.power_up_text_timer.isActive():
            self.power_up_text_timer.stop()
        
        self.update()
    
    def on_mode_changed(self):
        """Обработка изменения режима игры"""
        if self.normalModeRadio.isChecked():
            self.game.hard_mode = False
        else:
            self.game.hard_mode = True
        
        # Сбрасываем игру при смене режима
        self.game.reset_game()
        
        # Очищаем текст улучшений
        self.powerUpLabel.setText("")
        self.power_up_text = ""
        if self.power_up_text_timer.isActive():
            self.power_up_text_timer.stop()
        
        # Обновляем интерфейс
        self.update_ui()
        self.update()
        
    def on_algorithm_changed(self):
        """Обработка изменения алгоритма"""
        if self.astarRadio.isChecked():
            self.game.use_aco = False
        else:
            self.game.use_aco = True
        
        # Останавливаем игру при смене алгоритма
        if self.game.game_running:
            self.game_timer.stop()
            self.game.game_running = False
            self.startButton.setEnabled(True)
            self.pauseButton.setEnabled(False)
    
    def on_autoplay_changed(self, state):
        """Обработка изменения автопилота"""
        self.game.auto_play = (state == Qt.Checked)
        
        # Останавливаем игру при включении/выключении автопилота
        if self.game.game_running:
            self.game_timer.stop()
            self.game.game_running = False
            self.startButton.setEnabled(True)
            self.pauseButton.setEnabled(False)
    
    def game_tick(self):
        """Тик игрового таймера"""
        if self.game.auto_play and self.game.game_running and not self.game.game_paused:
            self.game.update_auto_play_direction()
        
        # Сохраняем предыдущий game_speed для проверки изменений
        prev_speed = self.game.game_speed
        
        # Двигаем змейку и проверяем, было ли применено улучшение
        power_up_applied = self.game.move_snake()
        
        # Если было применено улучшение, показываем сообщение
        if power_up_applied:
            power_up_text = self.game.get_power_up_text()
            if self.game.speed_mode:
                self.show_power_up_message(power_up_text, QColor(0, 0, 255))  # Синий
            elif self.game.ghost_mode:
                self.show_power_up_message(power_up_text, QColor(128, 0, 128))  # Фиолетовый
            elif self.game.slow_mode:
                self.show_power_up_message(power_up_text, QColor(128, 0, 0))  # Темно-красный
        
        # Обновляем улучшения
        if self.game.update_power_ups():
            # Сброс улучшения
            self.power_up_text = ""
            # Возвращаем нормальную скорость таймеру
            self.game_timer.setInterval(self.game.game_speed)
            # Останавливаем таймер анимации
            if self.power_up_text_timer.isActive():
                self.power_up_text_timer.stop()
        
        # Если скорость изменилась, обновляем интервал таймера
        if self.game.game_speed != prev_speed:
            self.game_timer.setInterval(self.game.game_speed)
        
        # Проверяем конец игры
        if self.game.game_over:
            self.game_timer.stop()
            self.gameOverLabel.setVisible(True)
            self.startButton.setEnabled(True)
            self.pauseButton.setEnabled(False)
            # Останавливаем таймер анимации
            if self.power_up_text_timer.isActive():
                self.power_up_text_timer.stop()
        
        if self.game.won:
            self.game_timer.stop()
            self.winLabel.setVisible(True)
            self.startButton.setEnabled(True)
            self.pauseButton.setEnabled(False)
            # Останавливаем таймер анимации
            if self.power_up_text_timer.isActive():
                self.power_up_text_timer.stop()
        
        self.update_ui()
        self.update()
    
    def generate_special_food_tick(self):
        """Генерация специального фрукта"""
        if self.game.game_running and not self.game.game_paused:
            if self.game.generate_special_food():
                self.update()
    
    def update_power_up_text(self):
        """Обновление анимации текста улучшений (как в .cpp)"""
        if self.game.is_power_up_active():
            # Двигаем текст вправо
            self.power_up_text_position += 3
            
            # Если текст ушел за пределы окна, сбрасываем позицию
            if self.power_up_text_position > self.width():
                self.power_up_text_position = 25
            
            self.update()
        else:
            # Останавливаем таймер, если улучшение закончилось
            self.power_up_text = ""
            self.power_up_text_timer.stop()
    
    def show_power_up_message(self, text: str, color: QColor):
        """Показать сообщение об улучшении (как в .cpp)"""
        self.power_up_text = text
        self.power_up_color = color
        self.power_up_text_position = 25
        
        # Запускаем таймер анимации
        if not self.power_up_text_timer.isActive():
            self.power_up_text_timer.start(50)
    
    def update_ui(self):
        """Обновление интерфейса"""
        # Обновляем счет
        self.scoreLabel.setText(f"Счет: {self.game.score} / {self.game.MAX_SCORE}")
        
        # Обновляем состояние кнопок
        if not self.game.game_running:
            self.startButton.setEnabled(True)
            self.pauseButton.setEnabled(False)
        
        # Обновляем состояние радиокнопок
        self.normalModeRadio.setChecked(not self.game.hard_mode)
        self.hardModeRadio.setChecked(self.game.hard_mode)
        
        self.astarRadio.setChecked(not self.game.use_aco)
        self.acoRadio.setChecked(self.game.use_aco)
        
        self.autoPlayCheckBox.setChecked(self.game.auto_play)
    
    def paintEvent(self, event):
        """Отрисовка игрового поля"""
        super().paintEvent(event)
        
        if not hasattr(self, 'game'):
            return
        
        # Получаем размеры виджета для игры
        game_widget = self.gameWidget
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Координаты и размеры игрового поля
        field_width = self.game.GRID_WIDTH * self.game.CELL_SIZE
        field_height = self.game.GRID_HEIGHT * self.game.CELL_SIZE
        field_left = game_widget.x() + (game_widget.width() - field_width) // 2
        field_top = game_widget.y() + (game_widget.height() - field_height) // 2
        
        # Рисуем фон
        painter.fillRect(field_left, field_top, field_width, field_height, QColor(0, 0, 0))
        
        # Рисуем сетку
        painter.setPen(QColor(128, 128, 128))
        for x in range(self.game.GRID_WIDTH + 1):
            painter.drawLine(
                field_left + x * self.game.CELL_SIZE,
                field_top,
                field_left + x * self.game.CELL_SIZE,
                field_top + field_height
            )
        
        for y in range(self.game.GRID_HEIGHT + 1):
            painter.drawLine(
                field_left,
                field_top + y * self.game.CELL_SIZE,
                field_left + field_width,
                field_top + y * self.game.CELL_SIZE
            )
        
        # Цвета змейки в зависимости от улучшений (как в .cpp)
        snake_color = QColor(0, 255, 0)  # Лаймовый
        head_color = QColor(0, 128, 0)   # Зеленый
        
        if self.game.speed_mode:
            snake_color = QColor(0, 0, 255)  # Синий
            head_color = QColor(0, 0, 128)   # Темно-синий
        elif self.game.ghost_mode:
            snake_color = QColor(128, 0, 128)  # Фиолетовый
            head_color = QColor(255, 0, 255)   # Фуксия
        elif self.game.slow_mode:
            snake_color = QColor(128, 0, 0)  # Темно-красный
            head_color = QColor(255, 0, 0)   # Красный
        
        # Рисуем змейку
        for i, segment in enumerate(self.game.snake):
            color = head_color if i == 0 else snake_color
            painter.fillRect(
                field_left + segment.x * self.game.CELL_SIZE + 1,
                field_top + segment.y * self.game.CELL_SIZE + 1,
                self.game.CELL_SIZE - 2,
                self.game.CELL_SIZE - 2,
                color
            )
        
        # Рисуем обычную еду
        painter.fillRect(
            field_left + self.game.food.x * self.game.CELL_SIZE + 1,
            field_top + self.game.food.y * self.game.CELL_SIZE + 1,
            self.game.CELL_SIZE - 2,
            self.game.CELL_SIZE - 2,
            QColor(255, 0, 0)
        )
        
        # Рисуем спецфрукт
        if self.game.special_food_active:
            if self.game.hard_mode:
                color = QColor(128, 0, 128)  # Фиолетовый (в сложном режиме все одинаковые)
            else:
                if self.game.special_food_type == 1:
                    color = QColor(0, 0, 255)  # Синий для ускорения
                elif self.game.special_food_type == 2:
                    color = QColor(128, 0, 128)  # Фиолетовый для призрака
                else:
                    color = QColor(128, 0, 0)  # Темно-красный для замедления
            
            painter.fillRect(
                field_left + self.game.special_food.x * self.game.CELL_SIZE + 1,
                field_top + self.game.special_food.y * self.game.CELL_SIZE + 1,
                self.game.CELL_SIZE - 2,
                self.game.CELL_SIZE - 2,
                color
            )
        
        # Рисуем текст улучшений (анимация бегущей строки)
        # Только если улучшение активно и есть текст
        if self.power_up_text and self.game.is_power_up_active():
            # Создаем жирный шрифт с увеличенным размером
            font = QFont("Arial", 16, QFont.Bold)
            painter.setFont(font)
            painter.setPen(self.power_up_color)
            
            # ЛУЧШАЯ ПОЗИЦИЯ: над игровым полем, но не слишком низко
            # Вычисляем позицию над игровым полем
            text_y = field_top + 480  # 30 пикселей выше игрового поля
            
            # Ограничиваем минимальную высоту, чтобы текст не уходил за верх окна
            if text_y < 40:
                text_y = 40
            
            # Рисуем бегущую строку
            painter.drawText(
                self.power_up_text_position,  # Горизонтальная позиция (бежит вправо)
                text_y,                       # Вертикальная позиция (фиксирована над полем)
                self.power_up_text
            )
    
    def keyPressEvent(self, event):
        """Обработка нажатий клавиш"""
        if not self.game.game_running or self.game.game_paused:
            super().keyPressEvent(event)
            return
        
        new_direction = self.game.direction
        
        key = event.key()
        # Обработка стрелок, WASD и русской раскладки ЦВЫФ
        if key in [Qt.Key_Up, Qt.Key_W, 0x0426]:  # Ц (W в русской раскладке)
            new_direction = Direction.UP
        elif key in [Qt.Key_Right, Qt.Key_D, 0x0412]:  # В (D в русской раскладке)
            new_direction = Direction.RIGHT
        elif key in [Qt.Key_Down, Qt.Key_S, 0x042B]:  # Ы (S в русской раскладке)
            new_direction = Direction.DOWN
        elif key in [Qt.Key_Left, Qt.Key_A, 0x0424]:  # Ф (A в русской раскладке)
            new_direction = Direction.LEFT
        else:
            super().keyPressEvent(event)
            return
        
        # Проверка разворота на 180 градусов
        if ((new_direction == Direction.UP and self.game.direction != Direction.DOWN) or
            (new_direction == Direction.DOWN and self.game.direction != Direction.UP) or
            (new_direction == Direction.RIGHT and self.game.direction != Direction.LEFT) or
            (new_direction == Direction.LEFT and self.game.direction != Direction.RIGHT)):
            self.game.next_direction = new_direction
        
        event.accept()