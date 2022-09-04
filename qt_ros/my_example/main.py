import os
import time
import sys
from collections.abc import Iterable
import math
import subprocess
import shlex
import signal
from loguru import logger
import yaml

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as SciRotation

from PyQt5.QtWidgets import QPushButton, QMainWindow, QApplication, QFileDialog, QTextEdit
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


from qt_ros.my_example.ui_example import Ui_MainWindow


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
            out, err = self._pipe.communicate(timeout=3)
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
    point_clouds_arrive_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.initialize_roscore()
        rospy.init_node('qt_ros_node')
        self.stop_flag = False
        self.replay_process = dict()
        self.rviz_process = []
        self.loop_rate = rospy.Rate(30, reset=True)

        self.turtlesim_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
        self.image_subscriber = rospy.Subscriber('/tracking_image', Image_msg, callback=self.image_callback, queue_size=1)
        self.pc_subscriber = rospy.Subscriber('/test_pointcloud', PointCloud2, callback=self.pc_callback, queue_size=1)
        self.image = None
        self.point_clouds = None

    def initialize_roscore(self):
        """
        roscore
        @return:
        """
        cmd = "ps -ef | grep 'roscore' | grep -v grep | awk '{print $2}'"
        old_pid = subprocess.getoutput(cmd)
        if old_pid == '':
            self.ros_process = subprocess.Popen('roscore', shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info('roscore initialized with pid %d' % self.ros_process.pid)
            time.sleep(0.5)
        else:
            logger.info('using already existed roscore process with pid %s' % old_pid)

    def run(self):
        """
        override
        """
        while not rospy.is_shutdown() and not self.stop_flag:
            self.loop_rate.sleep()

    def image_callback(self, msg):
        self.image = image_to_numpy(msg)
        self.image_arrive_signal.emit()

    def pc_callback(self, msg):
        self.point_clouds = cloud_msg2numpy(msg, fields=('x', 'y', 'z'))
        self.point_clouds_arrive_signal.emit()

    def turtlesim_publish(self):
        """
        rosrun turtlesim turtlesim_node
        rosrun turtlesim turtle_teleop_key
        @return:
        """
        sender = self.sender()
        twist_message = Twist()
        if sender.objectName() == 'up_btn':
            twist_message.linear = Vector3(2, 0, 0)
        elif sender.objectName() == 'down_btn':
            twist_message.linear = Vector3(-2, 0, 0)
        elif sender.objectName() == 'left_btn':
            twist_message.angular = Vector3(0, 0, 2)
        elif sender.objectName() == 'right_btn':
            twist_message.angular = Vector3(0, 0, -2)
        self.turtlesim_publisher.publish(twist_message)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.initialize_ui()
        self.ros_node = RosNode()
        self.ros_node.start()
        logger.info('ros node initialized.')
        self.refresh_topic()

        # self.shell_widget = ConsoleTextEdit(self)
        # self.shell_widget.setObjectName("shell_widget")
        # self.ui.verticalLayout_6.addWidget(self.shell_widget)

        # mainwindow
        self.ui.refresh_topic_btn.clicked.connect(self.refresh_topic)
        self.ui.topic_list.itemClicked.connect(self.show_topic_info)
        self.ui.record_bag_btn.clicked.connect(self.record_bag)
        self.ui.stop_record_bag_btn.clicked.connect(self.stop_record_bag)
        self.ui.play_btn.clicked.connect(self.play_bag)
        self.ui.stop_play_btn.clicked.connect(self.stop_play_bag)
        self.ui.stop_all_btn.clicked.connect(self.stop_all_bag)
        self.ui.show_bag_info_btn.clicked.connect(self.show_bag_info)
        self.ui.start_rviz_btn.clicked.connect(self.start_rviz)
        # turtlesim
        self.ui.up_btn.clicked.connect(self.ros_node.turtlesim_publish)
        self.ui.down_btn.clicked.connect(self.ros_node.turtlesim_publish)
        self.ui.left_btn.clicked.connect(self.ros_node.turtlesim_publish)
        self.ui.right_btn.clicked.connect(self.ros_node.turtlesim_publish)

        # ros
        self.ros_node.image_arrive_signal.connect(self.update_image)
        self.ros_node.point_clouds_arrive_signal.connect(self.update_point_clouds)

        # timer
        self.bag_process_timer = QTimer()
        self.bag_process_timer.timeout.connect(self.check_replay_process)
        self.bag_process_timer.start(300)

        self.rviz_process_timer = QTimer()
        self.rviz_process_timer.timeout.connect(self.check_rviz_process)
        self.rviz_process_timer.start(500)

        self.refresh_topic_timer = QTimer()

        # pyqtgraph
        self.grid_item = gl.GLGridItem()
        self.ui.og_widget.addItem(self.grid_item)

        self.x_axis_item = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32),
                                             color=(1, 0, 0, 1),
                                             width=2)
        self.ui.og_widget.addItem(self.x_axis_item)

        self.y_axis_item = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 10, 0]], dtype=np.float32),
                                             color=(0, 1, 0, 1),
                                             width=2)
        self.ui.og_widget.addItem(self.y_axis_item)

        self.z_axis_item = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, 10]], dtype=np.float32),
                                             color=(0, 0, 1, 1),
                                             width=2)
        self.ui.og_widget.addItem(self.z_axis_item)

    def initialize_ui(self):
        self.ui.stop_record_bag_btn.setEnabled(False)

    def update_image(self):
        width, height = self.ui.image_label.width(), self.ui.image_label.height()
        image = letterbox_image(self.ros_node.image, (height, width))
        raw_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        show_image = QImage(raw_image, raw_image.shape[1], raw_image.shape[0], raw_image.shape[1] * 3,
                            QImage.Format_RGB888)
        self.ui.image_label.setPixmap(QPixmap(show_image))

    def update_point_clouds(self):
        points = self.ros_node.point_clouds
        c = points[:, -1]
        colors = plt.get_cmap('jet')((c - c.min()) / (c.max() - c.min()))
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
            self.ui.og_widget.addItem(self.points_item)

    def start_rviz(self):
        """
        rviz
        @return:
        """
        rviz_process = subprocess.Popen('rviz', stdout=subprocess.PIPE)
        self.ros_node.rviz_process.append(rviz_process)
        logger.info('rviz started with pid %d' % rviz_process.pid)

    def check_rviz_process(self):
        for p in self.ros_node.rviz_process:
            if p.poll() is not None:
                logger.info('rviz with pid %d finished.' % p.pid)
                self.ros_node.rviz_process.remove(p)

    def refresh_topic(self):
        """
        rostopic list
        @return:
        """
        topics = subprocess.getoutput('rostopic list').split()
        self.ui.topic_list.clear()
        self.ui.topic_list.addItems(topics)

    def show_bag_info(self):
        """
        rosbag info
        @return:
        """
        path, ok = QFileDialog.getOpenFileName(self, '选择bag文件', './', 'Bag Files(*.bag)')
        if len(path) == 0:
            return
        cmd = 'rosbag info -y --freq %s' % path
        raw_bag_info = subprocess.getoutput(cmd)
        self.ui.bag_info.setText(raw_bag_info)
        bag_info = yaml.load(raw_bag_info, Loader=yaml.SafeLoader)
        return bag_info

    def show_topic_info(self, item):
        """
        rostopic info
        @param item:
        @return:
        """
        self.ui.topic_publisher.clear()
        self.ui.topic_subscribers.clear()
        cmd = 'rostopic info %s' % item.text()
        topic_info = subprocess.getoutput(cmd).split('\n')
        if topic_info[0].startswith('ERROR'):  # no this topic.
            self.refresh_topic()
            return
        topic_info = [i.strip() for i in topic_info if i.strip()]
        message_type = topic_info[0].split(':')[1].strip()

        if 'Publishers: None' not in topic_info:
            publisher = topic_info[2][2:]
            self.ui.topic_publisher.addItem(publisher)
        if 'Subscribers: None' not in topic_info:
            subscriber_idx = topic_info.index('Subscribers:')
            subscribers = topic_info[subscriber_idx + 1:]
            subscribers = [i[2:] for i in subscribers]
            self.ui.topic_subscribers.addItems(subscribers)

        self.ui.message_type.setText(message_type)

    def record_bag(self):
        """
        rosbag record
        @return:
        """
        cmd = 'rosbag record -a'
        self.record_bag_process = subprocess.Popen(shlex.split(cmd), shell=False, stdout=subprocess.PIPE)
        self.ui.record_bag_btn.setEnabled(False)
        self.ui.stop_record_bag_btn.setEnabled(True)
        logger.info('recording bag...')

    def stop_record_bag(self):
        """
        ctrl+c to rosbag record
        @return:
        """
        if hasattr(self, 'record_bag_process'):
            self.record_bag_process.send_signal(signal.SIGINT)
            self.ui.record_bag_btn.setEnabled(True)
            self.ui.stop_record_bag_btn.setEnabled(False)
            logger.info('finish recording bag.')
            del self.record_bag_process

    def play_bag(self):
        """
        rosbag play
        @return:
        """
        path, ok = QFileDialog.getOpenFileName(self, '选择bag文件', './', 'Bag Files(*.bag)')
        if len(path) != 0:
            bag_name = os.path.split(path)[-1]
            if bag_name in self.ros_node.replay_process:
                logger.info('bag %s is playing...')
                return
            cmd = 'rosbag play %s' % path
            p = subprocess.Popen(shlex.split(cmd), shell=False, stdout=subprocess.PIPE)
            self.ros_node.replay_process[bag_name] = p
            self.ui.playing_bag.addItem(bag_name)
            logger.info('playing bag %s...' % bag_name)
            self.refresh_topic_timer.singleShot(1000, self.refresh_topic)

    def stop_all_bag(self):
        """
        ctrl+c to all rosbag play
        @return:
        """
        for bag_name, p in self.ros_node.replay_process.items():
            p.send_signal(signal.SIGINT)
            p.wait()

    def stop_play_bag(self):
        """
        ctrl+c to rosbag play
        @return:
        """
        item = self.ui.playing_bag.currentItem()
        if item is not None:
            bag_name = item.text()
            p = self.ros_node.replay_process[bag_name]
            p.send_signal(signal.SIGINT)
            p.wait()

    def check_replay_process(self):
        """
        定时查看rosbag play process是否结束
        @return:
        """
        finished_bag_names = []
        for bag_name, p in self.ros_node.replay_process.items():
            if p.poll() is not None:
                finished_bag_names.append(bag_name)
                logger.info('finish playing bag %s.' % bag_name)
        if len(finished_bag_names) == 0:
            return

        for bag_name in finished_bag_names:
            self.ros_node.replay_process.pop(bag_name)
            for idx in range(self.ui.playing_bag.count()):
                item = self.ui.playing_bag.item(idx)
                if item.text() == bag_name:
                    self.ui.playing_bag.takeItem(idx)
                    break

        self.refresh_topic()

    def closeEvent(self, a0) -> None:
        self.ros_node.stop_flag = True
        for bag_name, p in self.ros_node.replay_process.items():
            p.send_signal(signal.SIGINT)
            p.wait()
        for p in self.ros_node.rviz_process:
            p.terminate()
        if hasattr(self.ros_node, 'ros_process'):
            self.ros_node.ros_process.send_signal(signal.SIGINT)
            self.ros_node.ros_process.wait()
            logger.info('roscore with pid %d finished.' % self.ros_node.ros_process.pid)
        self.ros_node.quit()
        self.ros_node.wait()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
