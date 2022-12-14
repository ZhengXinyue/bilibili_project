import sys

import numpy as np

from PyQt5.QtWidgets import QPushButton, QMainWindow, QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QIcon, QPixmap, QMouseEvent, QFont, QVector3D, QVector2D, QWindow
from PyQt5.QtCore import qDebug, QTimer, QObject, QEvent, Qt, pyqtSignal, QProcess, QSettings

import pyqtgraph.opengl as gl


class App(QMainWindow):
    def __init__(self):
        super(App, self).__init__()
        self.resize(1600, 900)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setWindowTitle('3D_visualizer')
        self.widget = QWidget()
        self.setCentralWidget(self.widget)

        # layout
        self.layout = QHBoxLayout(self.widget)
        self.vis_layout = QHBoxLayout()
        self.control_layout = QVBoxLayout()
        self.layout.addLayout(self.control_layout)
        self.layout.addLayout(self.vis_layout)
        self.layout.setStretch(0, 1)
        self.layout.setStretch(1, 8)

        self.pg_widget = gl.GLViewWidget(parent=self.widget)
        self.pg_widget.addItem(gl.GLAxisItem(size=QVector3D(10, 10, 10)))
        self.vis_layout.addWidget(self.pg_widget)

        self.add_btn = QPushButton('add')
        self.control_layout.addWidget(self.add_btn)

        self.add_btn.clicked.connect(self.add_point)
        self.point_pos = np.array([[0, 0, 0]])

        # 生成wid
        with open('winId.txt', 'w') as f:
            f.write(str(int(self.winId())))

    def add_point(self):
        item = gl.GLScatterPlotItem(pos=self.point_pos, size=np.array([3]), pxMode=False)
        self.pg_widget.addItem(item)
        self.point_pos += 5


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec_())
