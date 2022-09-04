import os
import time
import sys
from collections.abc import Iterable
import math
import subprocess
import shlex
import signal

import pyqtgraph.opengl
from loguru import logger
import yaml

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as SciRotation

from PyQt5.QtWidgets import QPushButton, QMainWindow, QApplication, QFileDialog, QWidget, QVBoxLayout, QLabel, \
    QSizePolicy, QDockWidget, QTextEdit
from PyQt5.QtGui import QIcon, QPixmap, QMouseEvent, QFont, QVector3D, QImage
from PyQt5.QtCore import qDebug, QTimer, QObject, QEvent, Qt, pyqtSignal, QThread, QSize

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph import Transform3D, ColorMap, mkColor, makeRGBA

import rospy
from std_msgs.msg import String, Int8
from sensor_msgs.msg import JointState, Imu, Image as Image_msg, PointCloud2
from ros_numpy.image import image_to_numpy, numpy_to_image
from ros_numpy.point_cloud2 import pointcloud2_to_array
from geometry_msgs.msg import Twist, Vector3
from rostopic import _rostopic_list, get_info_text


def cloud_msg2numpy(cloud_msg, fields=('x', 'y', 'z', 'intensity'), max_intensity=float('inf'), remove_nans=True):
    """
    从ros的雷达原始消息中获取相应字段信息
    :param cloud_msg:       PointCloud2   ros消息类型
    :param fields:          tuple  需要的fields
    :param remove_nans:     bool
    :param max_intensity:   int   最大强度阈值   ouster的雷达强度从0到3000+且数据分布不均匀, 造成大部分数据的范围相对压缩.
    :return:                points: numpy.ndarray `N x len(fields)`
    """
    cloud_array = pointcloud2_to_array(cloud_msg)
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]
    # pull out target fields
    points = np.zeros(cloud_array.shape + (len(fields),), dtype=np.float32)

    for i, field_name in enumerate(fields):
        points[..., i] = cloud_array[field_name]
        if field_name == 'intensity':
            points[..., i] = np.minimum(max_intensity, points[..., i])

    return points


def letterbox_image(image, target_shape):
    """
    缩放图片, 填充短边
    :param image:            np.ndarray [H, W, C]
    :param target_shape:     tuple (H, W)
    :return:
    """
    image_h, image_w = image.shape[:2]
    target_h, target_w = target_shape
    # 获取缩放尺度, resize
    scale = min(float(target_h) / image_h, float(target_w) / image_w)
    new_h = int(image_h * scale)
    new_w = int(image_w * scale)
    image = cv2.resize(image, (new_w, new_h))
    canvas = np.zeros(shape=[target_h, target_w, 3], dtype=np.float32)
    canvas[:, :, :] = (128, 128, 128)
    start_h, start_w = (target_h - new_h) // 2, (target_w - new_w) // 2
    canvas[start_h:start_h + new_h, start_w:start_w + new_w, :] = image[:, :, :]
    canvas = canvas.astype(np.uint8)
    return canvas


class ConsoleTextEdit(QTextEdit):
    _color_stdout = Qt.blue
    _color_stderr = Qt.red
    _color_stdin = Qt.black
    _multi_line_char = '\\'
    _multi_line_indent = '    '
    _prompt = ('$ ', '  ')  # prompt for single and multi line

    class TextEditColoredWriter:

        def __init__(self, text_edit, color):
            self._text_edit = text_edit
            self._color = color

        def write(self, line):
            old_color = self._text_edit.textColor()
            self._text_edit.setTextColor(self._color)
            self._text_edit.insertPlainText(line)
            self._text_edit.setTextColor(old_color)
            self._text_edit.ensureCursorVisible()

    def __init__(self, parent=None):
        super(ConsoleTextEdit, self).__init__(parent)
        self.setFont(QFont('Mono'))

        self._multi_line = False
        self._multi_line_level = 0
        self._command = ''
        self._history = []
        self._history_index = -1

        # init colored writers
        self._stdout = self.TextEditColoredWriter(self, self._color_stdout)
        self._stderr = self.TextEditColoredWriter(self, self._color_stderr)
        self._comment_writer = self.TextEditColoredWriter(self, self._color_stdin)
        self._add_prompt()

    def print_message(self, msg):
        self._clear_current_line(clear_prompt=True)
        self._comment_writer.write(msg + '\n')
        self._add_prompt()

    def _add_prompt(self):
        self._comment_writer.write(
            self._prompt[self._multi_line] + self._multi_line_indent * self._multi_line_level)

    def _clear_current_line(self, clear_prompt=False):
        # block being current row
        prompt_length = len(self._prompt[self._multi_line])
        if clear_prompt:
            prompt_length = 0
        length = len(self.document().lastBlock().text()[prompt_length:])
        if length == 0:
            return None
        else:
            # should have a better way of doing this but I can't find it
            for _ in range(length):
                self.textCursor().deletePreviousChar()
        return True

    def _move_in_history(self, delta):
        # used when using the arrow keys to scroll through _history
        self._clear_current_line()
        if -1 <= self._history_index + delta < len(self._history):
            self._history_index += delta
        if self._history_index >= 0:
            self.insertPlainText(self._history[self._history_index])
        return True

    def _exec_code(self, code):
        try:
            self._pipe = subprocess.Popen([code], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = self._pipe.communicate(timeout=3)   # 防止永远阻塞
            self._stdout.write(out.decode('utf-8'))
            self._stderr.write(err.decode('utf-8'))
        except Exception as e:
            self._stderr.write(str(e) + '\n')

    def keyPressEvent(self, event):
        prompt_length = len(self._prompt[self._multi_line])
        block_length = self.document().lastBlock().length()
        document_length = self.document().characterCount()
        line_start = document_length - block_length
        prompt_position = line_start + prompt_length

        # only handle keys if cursor is in the last line
        if self.textCursor().position() >= prompt_position:
            if event.key() == Qt.Key_Down:
                if self._history_index == len(self._history):
                    self._history_index -= 1
                self._move_in_history(-1)
                return None

            if event.key() == Qt.Key_Up:
                self._move_in_history(1)
                return None

            if event.key() in [Qt.Key_Backspace]:
                # don't allow cursor to delete into prompt
                if (self.textCursor().positionInBlock() == prompt_length and
                        not self.textCursor().hasSelection()):
                    return None

            if event.key() in [Qt.Key_Return, Qt.Key_Enter]:
                # set cursor to end of line to avoid line splitting
                cursor = self.textCursor()
                cursor.setPosition(document_length - 1)
                self.setTextCursor(cursor)

                self._history_index = -1
                line = str(self.document().lastBlock().text())[
                    prompt_length:].rstrip()  # remove prompt and trailing spaces

                self.insertPlainText('\n')
                if len(line) > 0:
                    if line[-1] == self._multi_line_char:
                        self._multi_line = True
                        self._multi_line_level += 1
                    self._history.insert(0, line)

                    if self._multi_line:  # multi line command
                        self._command += line + '\n'

                    else:  # single line command
                        self._exec_code(line)
                        self._command = ''

                else:  # new line was is empty

                    if self._multi_line:  # multi line done
                        self._exec_code(self._command)
                        self._command = ''
                        self._multi_line = False
                        self._multi_line_level = 0

                self._add_prompt()
                return None

        # allow all other key events
        super(ConsoleTextEdit, self).keyPressEvent(event)

        # fix cursor position to be after the prompt, if the cursor is in the last line
        if line_start <= self.textCursor().position() < prompt_position:
            cursor = self.textCursor()
            cursor.setPosition(prompt_position)
            self.setTextCursor(cursor)


class RosNode(QThread):
    data_arrive_signal = pyqtSignal()
    image_arrive_signal = pyqtSignal()
    pointcloud_arrive_signal = pyqtSignal()

    def __init__(self):
        super(RosNode, self).__init__()
        self.initialize_roscore()
        rospy.init_node('qt_ros_node')
        self.stop_flag = False

        self.loop_rate = rospy.Rate(30, reset=True)
        self.turtlesim_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=1)
        self.turtlesim_subscriber = rospy.Subscriber('/turtle1/cmd_vel', Twist, callback=self.f, queue_size=1)
        self.image_subscriber = rospy.Subscriber('/tracking_image', Image_msg, callback=self.image_callback,
                                                 queue_size=1)
        self.pointcloud_subscriber = rospy.Subscriber('/test_pointcloud', PointCloud2,
                                                      callback=self.pointcloud_callback, queue_size=1)

        self.msg = None
        self.image = None
        self.points = None

    def initialize_roscore(self):
        """
        initialize the roscore process.
        :return:
        """
        cmd = "ps -ef | grep 'roscore' | grep -v grep | awk '{print $2}'"
        old_pid = subprocess.getoutput(cmd)
        if old_pid == '':
            self.roscore_process = subprocess.Popen(shlex.split('roscore'),
                                                    stdout=subprocess.PIPE,
                                                    stderr=subprocess.PIPE,
                                                    shell=False)
            logger.info('roscore initialized with pid %d' % self.roscore_process.pid)
            time.sleep(0.5)
        else:
            logger.info('using already existed roscore process with pid %s' % old_pid)

    def pointcloud_callback(self, msg):
        self.points = cloud_msg2numpy(msg, fields=('x', 'y', 'z'))
        self.pointcloud_arrive_signal.emit()

    def image_callback(self, msg):
        self.image = image_to_numpy(msg)
        self.image_arrive_signal.emit()

    def f(self, msg):
        self.msg = msg
        self.data_arrive_signal.emit()

    def publish_twist_message(self):
        sender = self.sender()
        message = Twist()
        if sender.text() == 'up':
            message.linear = Vector3(2, 0, 0)
        elif sender.text() == 'down':
            message.linear = Vector3(-2, 0, 0)
        elif sender.text() == 'left':
            message.angular = Vector3(0, 0, 2)
        elif sender.text() == 'right':
            message.angular = Vector3(0, 0, -2)
        self.turtlesim_publisher.publish(message)

    def run(self) -> None:
        while not rospy.is_shutdown() and not self.stop_flag:
            self.loop_rate.sleep()


class MainWindow(QMainWindow):
    """
    在pyqt中进行三维数据可视化(以激光雷达点云为例, mayavi, open3d, pyqtgraph, opengl, vtk等方法解析)
    """

    def __init__(self):
        super(MainWindow, self).__init__()
        self.resize(1600, 900)
        self.widget = QWidget()
        self.setCentralWidget(self.widget)
        self.layout = QVBoxLayout(self.widget)

        self.og_widget = gl.GLViewWidget()
        self.layout.addWidget(self.og_widget)
        self.grid_item = gl.GLGridItem()
        self.og_widget.addItem(self.grid_item)

        self.x_axis_item = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32),
                                             color=(1, 0, 0, 1),
                                             width=2)
        self.og_widget.addItem(self.x_axis_item)

        self.y_axis_item = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 10, 0]], dtype=np.float32),
                                             color=(0, 1, 0, 1),
                                             width=2)
        self.og_widget.addItem(self.y_axis_item)

        self.z_axis_item = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, 10]], dtype=np.float32),
                                             color=(0, 0, 1, 1),
                                             width=2)
        self.og_widget.addItem(self.z_axis_item)

        self.up_btn = QPushButton('up', self.widget)
        self.down_btn = QPushButton('down')
        self.left_btn = QPushButton('left')
        self.right_btn = QPushButton('right')
        self.label = QLabel('Data to show')
        self.layout.addWidget(self.up_btn)
        self.layout.addWidget(self.down_btn)
        self.layout.addWidget(self.left_btn)
        self.layout.addWidget(self.right_btn)
        self.layout.addWidget(self.label)

        self.add_item_btn = QPushButton('add')
        self.delete_item_btn = QPushButton('delete')
        self.layout.addWidget(self.add_item_btn)
        self.layout.addWidget(self.delete_item_btn)
        self.add_item_btn.clicked.connect(self.add_item)
        self.delete_item_btn.clicked.connect(self.delete_item)

        self.ros_topic_list_btn = QPushButton('topic list')
        self.layout.addWidget(self.ros_topic_list_btn)
        self.ros_topic_list_btn.clicked.connect(self.show_topic_list)

        self.start_rviz_btn = QPushButton('rviz')
        self.layout.addWidget(self.start_rviz_btn)
        self.start_rviz_btn.clicked.connect(self.start_rviz)

        self.start_bag_record_btn = QPushButton('start record')
        self.stop_bag_record_btn = QPushButton('stop record')
        self.layout.addWidget(self.start_bag_record_btn)
        self.layout.addWidget(self.stop_bag_record_btn)
        self.start_bag_record_btn.clicked.connect(self.start_bag_record)
        self.stop_bag_record_btn.clicked.connect(self.stop_bag_record)

        self.console_text_edit = ConsoleTextEdit()
        self.layout.addWidget(self.console_text_edit)
        self.layout.setStretch(0, 7)

        self.image_label = QLabel('image')
        self.image_label.setMinimumSize(300, 1)
        size_policy = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setSizePolicy(size_policy)
        self.image_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        self.dock_widget = QDockWidget(self)
        self.dock_widget.setWindowTitle('Image')
        self.dock_widget.setWidget(self.image_label)
        self.addDockWidget(Qt.DockWidgetArea(1), self.dock_widget)

        self.image_label2 = QLabel('image2')
        self.image_label2.setMinimumSize(300, 1)
        size_policy = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label2.setSizePolicy(size_policy)
        self.image_label2.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.dock_widget2 = QDockWidget(self)
        self.dock_widget2.setWindowTitle('Image2')
        self.dock_widget2.setWidget(self.image_label2)
        self.addDockWidget(Qt.DockWidgetArea(1), self.dock_widget2)

        self.ros_node = RosNode()
        self.ros_node.start()

        self.up_btn.clicked.connect(self.ros_node.publish_twist_message)
        self.down_btn.clicked.connect(self.ros_node.publish_twist_message)
        self.left_btn.clicked.connect(self.ros_node.publish_twist_message)
        self.right_btn.clicked.connect(self.ros_node.publish_twist_message)
        self.ros_node.data_arrive_signal.connect(self.show_msg)
        self.ros_node.pointcloud_arrive_signal.connect(self.show_pointcloud)

        self.ros_node.image_arrive_signal.connect(self.show_image)

    def start_bag_record(self):
        cmd = 'rosbag record -a'
        self.ros_node.record_process = subprocess.Popen(shlex.split(cmd),
                                                        stdout=subprocess.PIPE,
                                                        stderr=subprocess.PIPE,
                                                        shell=False)

    def stop_bag_record(self):
        self.ros_node.record_process.send_signal(signal.SIGINT)

    def start_rviz(self):
        cmd = 'rviz'
        self.ros_node.rviz_process = subprocess.Popen(shlex.split(cmd),
                                                      stdout=subprocess.PIPE,
                                                      stderr=subprocess.PIPE,
                                                      shell=False)
        logger.info('rviz started with pid %d' % self.ros_node.rviz_process.pid)

    def show_topic_list(self):
        cmd = 'rostopic list'
        output = subprocess.getoutput(cmd)
        print(output)
        # _rostopic_list(None)
        # text = get_info_text('/rosout')
        # print(text)

    def add_item(self):
        self.box_item = gl.GLScatterPlotItem(pos=np.array([10, 10, 10]),
                                             color=(1, 0, 0, 1),
                                             size=5,
                                             pxMode=False)
        self.og_widget.addItem(self.box_item)

    def delete_item(self):
        if hasattr(self, 'box_item'):
            self.og_widget.removeItem(self.box_item)
            del self.box_item

    def show_pointcloud(self):
        points = self.ros_node.points
        colors = np.ones(shape=(points.shape[0], 4))
        size = np.zeros(shape=points.shape[0]) + 0.03
        if hasattr(self, 'points_item'):
            self.points_item.setData(pos=points,
                                     color=colors,
                                     size=size,
                                     pxMode=False)
        else:
            self.points_item = gl.GLScatterPlotItem(pos=points,
                                                    color=colors,
                                                    size=size,
                                                    pxMode=False)
            self.og_widget.addItem(self.points_item)

    def show_image(self):
        image = cv2.cvtColor(self.ros_node.image, cv2.COLOR_BGR2RGB)
        target_shape1 = [self.dock_widget.height(), self.dock_widget.width()]
        image1 = letterbox_image(image, target_shape1)
        show_image1 = QImage(image1, image1.shape[1], image1.shape[0], image1.shape[1] * 3, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap(show_image1))

        target_shape2 = [self.dock_widget2.height(), self.dock_widget2.width()]
        image2 = letterbox_image(image, target_shape2)
        show_image2 = QImage(image2, image2.shape[1], image2.shape[0], image2.shape[1] * 3, QImage.Format_RGB888)
        self.image_label2.setPixmap(QPixmap(show_image2))

    def show_msg(self):
        self.label.setText(str(self.ros_node.msg.linear.x))

    def closeEvent(self, a0) -> None:
        self.ros_node.stop_flag = True
        if hasattr(self.ros_node, 'rviz_process'):
            self.ros_node.rviz_process.send_signal(signal.SIGINT)
            self.ros_node.rviz_process.wait()
            logger.info('rviz with pid %d finished.' % self.ros_node.rviz_process.pid)
        if hasattr(self.ros_node, 'roscore_process'):
            self.ros_node.roscore_process.send_signal(signal.SIGINT)
            self.ros_node.roscore_process.wait()
            logger.info('roscore with pid %d finished.' % self.ros_node.roscore_process.pid)
        self.ros_node.quit()
        self.ros_node.wait()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    # apply_stylesheet(app, theme='dark_teal.xml')
    w.show()
    sys.exit(app.exec_())
