import sys

from win32gui import FindWindowEx

from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QHBoxLayout
from PyQt5.QtGui import QWindow


class MainApp(QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.resize(1600, 900)
        self.setWindowTitle('new window')
        self.widget = QWidget()
        self.setCentralWidget(self.widget)
        self.layout = QHBoxLayout(self.widget)

        # 读取wid
        with open('winId.txt', 'r') as f:
            win_id = int(f.readline())

        # win_id = int(FindWindowEx(0, 0, 0, '3D_Visualizer'))
        print('get winid %d' % win_id)
        self.child_window = QWindow.fromWinId(win_id)
        self.child_widget = QWidget.createWindowContainer(self.child_window, self)
        self.layout.addWidget(self.child_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainApp()
    w.show()
    sys.exit(app.exec_())
