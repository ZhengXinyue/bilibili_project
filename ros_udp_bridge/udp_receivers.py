from collections import defaultdict
from copy import deepcopy

from ros_udp_bridge.ros_udp import *


def parse_data(message):
    data_dict = defaultdict(list)

    while len(message) >= 6:
        head_frame = message[:2]

        if head_frame not in udp_definition:
            # print('Unknow head frame: ', head_frame.hex())
            message = message[2:]
            continue
        length_frame = message[2:6]
        try:
            data_length = struct.unpack('<I', length_frame)[0]
        except:
            print('Length frame parse error: ', length_frame.hex())
            message = message[2:]
            continue
        curr_message = message[:data_length]

        if len(curr_message) < data_length:
            # print('Not enough message')
            return data_dict, message

        tail_frame = curr_message[-1:]
        data_type, tail_definition = udp_definition[head_frame]
        if tail_frame != tail_definition:
            print('Inconsistent tail %s and %s' % (tail_frame.hex(), tail_definition.hex()))
            message = message[2:]
            continue
        data_frame = curr_message[6:-1]
        if data_type == UINT8_MSG:
            uint8_data = struct.unpack('<B', data_frame)[0]
            data_dict[data_type].append(uint8_data)
        elif data_type == FLOAT32_MSG:
            age = struct.unpack('<B', data_frame[:1])[0]
            weight = struct.unpack('<f', data_frame[1:5])[0]
            height = struct.unpack('<H', data_frame[5:7])[0]
            data_dict[data_type].append([age, weight, height])
        elif data_type == PC2_MSG:
            pc2 = np.frombuffer(data_frame, dtype=np.float32).reshape((-1, 3))
            data_dict[data_type].append(pc2)

        message = message[data_length:]

    return data_dict, message


def f(m):
    return m


class UdpReceiver(object):
    def __init__(self):
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.data_socket.bind(('', 8005))

    def start_listen(self):
        prev_message = b''
        while True:
            message, address = self.data_socket.recvfrom(8192)
            print(len(message))
            prev_message = f(prev_message + message)

            # parsed_data, prev_message = parse_data(prev_message + message)
            # if PC2_MSG in parsed_data:
            #     print(parsed_data)


if __name__ == '__main__':
    # receiver = UdpReceiver()
    # receiver.start_listen()

    data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data_socket.bind(('', 8005))

    prev_message = b''
    while True:
        message, address = data_socket.recvfrom(8192)
        print(len(message))
        # prev_message = f(prev_message + message)
