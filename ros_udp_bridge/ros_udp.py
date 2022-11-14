import socket
import struct
import time
from threading import Lock

import numpy as np

import rospy
from std_msgs.msg import Float32MultiArray, UInt8
from sensor_msgs.msg import PointCloud2

from ros_numpy.point_cloud2 import pointcloud2_to_array, get_xyz_points


# head frame definition and tail frame definition
UINT8_MSG = 0
FLOAT32_MSG = 1
PC2_MSG = 2

ros_udp_definition = {
    UINT8_MSG: (bytes.fromhex('fef1'), bytes.fromhex('1f')),           # (head frame, tail frame)
    FLOAT32_MSG: (bytes.fromhex('fef2'), bytes.fromhex('2f')),
    PC2_MSG: (bytes.fromhex('fef3'), bytes.fromhex('3f')),
}

udp_definition = {v[0]: (k, v[1]) for k, v in ros_udp_definition.items()}     # {head_message: (data_type, tail_message)}
head_messages_set = set(udp_definition.keys())
print('udp_definition: ', {k.hex(): (v[0], v[1].hex()) for k, v in udp_definition.items()})
print('head_message_set: ', [i.hex() for i in head_messages_set])


class RosUdpBridge(object):
    def __init__(self):
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.data_socket.bind(('127.0.0.1', 8000))

        self.target_port = ('127.0.0.1', 8005)

        rospy.init_node('ros_subscribers')
        self.pc2_subscriber = rospy.Subscriber('/pc2_topic', PointCloud2, queue_size=1, callback=self.pc2_callback)
        self.uint8_subscriber = rospy.Subscriber('/uint8_topic', UInt8, queue_size=1, callback=self.uint8_callback)
        self.float32_ma_subscriber = rospy.Subscriber('/float32_ma_topic', Float32MultiArray, queue_size=1, callback=self.float32_ma_callback)

        self.lock = Lock()

    def pc2_callback(self, msg):
        data = pointcloud2_to_array(msg)
        data = get_xyz_points(data)   # N x 3
        # data = data[:5, :]

        head_frame, tail_frame = ros_udp_definition[PC2_MSG]

        data_frame = data.astype(np.float32).tobytes()

        data_length = 2 + 4 + len(data_frame) + 1
        length_frame = struct.pack('<I', data_length)

        udp_msg = b''.join([head_frame, length_frame, data_frame, tail_frame])

        slice_count = len(udp_msg) // 1024 + 1
        sliced_data = [udp_msg[i * 1024: (i + 1) * 1024] for i in range(slice_count)]
        with self.lock:
            for i in sliced_data:
                self.data_socket.sendto(i, self.target_port)
                time.sleep(0.001)   # 避免发送频率过快丢包

    def uint8_callback(self, msg):
        data = msg.data   # uint8
        head_frame, tail_frame = ros_udp_definition[UINT8_MSG]
        data_frame = struct.pack('<B', data)

        data_length = 2 + 4 + len(data_frame) + 1
        length_frame = struct.pack('<I', data_length)

        udp_msg = b''.join([head_frame, length_frame, data_frame, tail_frame])

        with self.lock:
            self.data_socket.sendto(udp_msg, self.target_port)

    def float32_ma_callback(self, msg):
        data = msg.data
        head_frame, tail_frame = ros_udp_definition[FLOAT32_MSG]

        # age      uint8
        # weight   float32
        # height   u short 16
        data_frame = b''.join([struct.pack('<B', int(data[0])),
                               struct.pack('<f', data[1]),
                               struct.pack('<H', int(data[2]))])

        data_length = 2 + 4 + len(data_frame) + 1
        length_frame = struct.pack('<I', data_length)

        udp_msg = b''.join([head_frame, length_frame, data_frame, tail_frame])

        with self.lock:
            self.data_socket.sendto(udp_msg, self.target_port)

    def build_message(self, head_frame, data_frame, tail_frame):
        """
        build and send
        """
        pass


if __name__ == '__main__':
    bridge = RosUdpBridge()
    rospy.spin()
