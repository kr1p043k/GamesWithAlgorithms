import sys
from PyQt5.QtWidgets import QApplication
from game_window import SnakeGameWindow

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = SnakeGameWindow()
    window.setFixedSize(800, 625) 
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()