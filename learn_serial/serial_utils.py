import re
import struct
from collections.abc import Iterable
from collections import defaultdict

import serial
import serial.tools.list_ports


def get_ports():
    port_list = serial.tools.list_ports.comports()
    return [i.device for i in port_list]


def open_port(port_name, baudrate, bytesize=serial.EIGHTBITS, stopbits=serial.STOPBITS_ONE, parity=serial.PARITY_NONE):
    ser = serial.Serial(port=port_name,
                        baudrate=baudrate,
                        bytesize=bytesize,
                        stopbits=stopbits,
                        parity=parity,
                        rtscts=False,
                        xonxoff=False,
                        timeout=None,
                        write_timeout=None)
    return ser


def object2hex(x, codes):
    """
    2.44 ——> f6281c40
    45 ——> 2d000000
    :param x:       float
    :param codes:   str
    :return:        str
    <是小端  >是大端  !network(大端)  =本机
    f代表float  I代表unsigned int  i代表int  d代表double
    """
    return struct.pack(codes, x).hex()


def object2bytes(x, codes):
    """
    2.44 ——> b'\xf6(\x1c@'
    45 ——> b'\x00\x004B'
    可以直接send to serial
    :param x:        float
    :param codes:    str
    :return:         bytes
    """
    return struct.pack(codes, x)


def hex2object(x, codes):
    """
    f6281c40——>2.44
    :param x:        str
    :param codes:    str
    :return:         float
    """
    return struct.unpack(codes, bytes.fromhex(x))[0]


def bytes2object(x, codes):
    """
    b'\xf6(\x1c@'——>2.44
    :param x:       bytes
    :param codes:   bytes
    :return:        object
    """
    return struct.unpack(codes, x)[0]


def bytes2hex(x):
    """
    b'\xf6(\x1c@'——>f6281c40
    :param x:   bytes
    :return:    str
    """
    return x.hex()


def hex2bytes(x):
    """
    f6281c40——>b'\xf6(\x1c@'
    :param x:    str
    :return:     bytes
    """
    return bytes.fromhex(x)


def build_message(head_frame, data_frame, tail_frame, codes, first_recursion=True):
    """
    构造二进制消息
    :param head_frame:  帧头
        str: 55 DA
    :param data_frame:
        str: AA AA AA AA AA 68 11 04 34 37 33 37
        bytes: b'\xaa\xaa\xaa\xaa\xaah\x11\x044737'
        float: 2.44
        int:   34
        Iterable: [2.44, 3.12, 3.55]
    :param tail_frame:  帧尾
        str: specified or empty
    :param first_recursion:
        add end frame if True
    :param codes:
        encoding for struck.pack()   eg: f for float32    4
                                         B for uint8      1
                                         b for int 8      1
                                         i for int32      4
                                         I for uint32     4
                                         d for double     8
    :return:
    """
    head_frame = hex2bytes(head_frame)
    tail_frame = hex2bytes(tail_frame)
    if isinstance(data_frame, str):   # str
        bytes_data = bytes.fromhex(data_frame)
    elif isinstance(data_frame, bytes):   # bytes
        bytes_data = data_frame
    elif isinstance(data_frame, Iterable):
        bytes_data = b''.join([build_message(head_frame='', data_frame=i, tail_frame='', codes=codes, first_recursion=False) for i in data_frame])
    else:
        bytes_data = object2bytes(data_frame, codes=codes)
    if first_recursion:
        # computer bytes_data sum  数据位求和
        pass
    return b''.join([head_frame, bytes_data, tail_frame])


def parse_one_frame(message, total_length, codes, head_length=2, tail_length=1):
    """
    解析一帧数据
    :param message:
        bytes: b'U\xda\xf6(\x1c@\x14\xaeG@33c@\xb6'
        str:   55d7d7a3f03eff
    :param total_length:   int 一帧消息的字节数量
        int:
    :param tail_length:    int 帧头的字节数
    :param head_length:    int 帧尾的字节数
    :param codes:          encoding for struct.unpack()
        int:
    :return:
        head_frame:   str:                         55da
        data:         list[object, object, ...]    [2.44, 3.12, 3.55]
        tail_frame:   str                          b6
    """
    if isinstance(message, bytes):
        message = bytes2hex(message)   # str
    assert len(message) == total_length * 2
    head_frame = message[:head_length * 2]   # str
    tail_frame = message[-tail_length * 2:]    # str
    data_frame = message[head_length * 2: -tail_length*2]  # str
    data_list = re.findall('.{8}', data_frame)   # 一个float or int数据四个字节, 等长划分字符串 TODO
    data = [hex2object(i, codes=codes) for i in data_list]
    return head_frame, data, tail_frame


POINT_CLOUDS = 0    # 雷达点云
POSITION_X = 1      # 无人机位置X
POSITION_Y = 2      # 无人机位置Y
POSITION_Z = 3      # 无人机位置Z
POSTURE_X = 4       # 无人机姿态X
POSTURE_Y = 5       # 无人机姿态Y
POSTURE_Z = 6       # 无人机姿态Z
POSTURE_W = 7       # 无人机姿态W

serial_definition = {
    # 'head_frame': (total_length, 'tail_frame', data_type)
    '55da': (16, 'ad', POINT_CLOUDS),
    '55d1': (8, '1d', POSTURE_X),
    '55d2': (8, '2d', POSTURE_Y),
    '55d3': (8, '3d', POSTURE_Z),
    '55d4': (8, '4d', POSTURE_W),
    '55d5': (8, '5d', POSITION_X),
    '55d6': (8, '6d', POSITION_Y),
    '55d7': (8, '7d', POSITION_Z),
}


def parse_multi_frame(message, head_length, tail_length, codes, debug=False):
    """
    解析多帧数据
    :param message:
        bytes: b'U\xda\xf6(\x1c@\x14\xaeG@33c@\xb6'
        str:   55d7d7a3f03eff
    :param head_length:    int 帧尾的字节数
    :param tail_length:    int 帧头的字节数
    :param codes:          encoding for struct.unpack()  '<f' '<i' ...
    :param debug:          bool
    :return:
        return_data:   dict()  {1: [[3.55]],
                                0: [[3.45, 3.55, 1.77], [1.33, 5.77, 2.66]], ...}
        lefted_message:   bytes  剩余的消息, 放到下一次解析
    """
    if isinstance(message, bytes):
        message = bytes2hex(message)   # str
    return_data = defaultdict(list)   # store data by head frame
    current_idx = 0
    while current_idx < len(message):
        head_frame = message[current_idx:current_idx+head_length*2]
        if head_frame not in serial_definition:
            if debug:
                print("Parse error: head frame %s doesn't exist in serial definition" % head_frame)
            current_idx += 2   # go to next bytes and try again
            continue
        message_length, tail_frame, data_type = serial_definition[head_frame]   # get message length by definition
        curr_message = message[current_idx:current_idx+message_length * 2]
        # print(curr_message, message_length, head_length, tail_length)
        # print(len(curr_message), current_idx, message_length * 2)
        if len(curr_message) < message_length * 2:
            return return_data, bytes.fromhex(curr_message)
        _, data, message_tail_frame = parse_one_frame(curr_message, total_length=message_length, codes=codes, head_length=head_length, tail_length=tail_length)
        if message_tail_frame[2:].lower() != tail_frame.lower():   # check sum and real tail frame
            print("Parse error: tail frame %s doesn't equal to tail frame definition %s" % (message_tail_frame[2:], tail_frame))
            current_idx += 2  # go to next bytes and try again
            continue
        return_data[data_type].append(data)
        current_idx = current_idx + message_length * 2

    return return_data, b''


def message_send_receive_test():
    x, y, z = 4.66, 3.55, 5.12
    hex_x = object2hex(x, codes='<f')
    print('hex_x: ', hex_x)
    bytes_x = object2bytes(x, codes='<f')
    print('bytes_x: ', bytes_x)

    start_frame = '55 DA'
    end_frame = 'B6'
    data_send = b''.join([hex2bytes(start_frame), object2bytes(x, codes='<f'), hex2bytes(end_frame)])
    # ------------------------------------------------#
    data_receive = data_send
    print('data_receive_bytes: ', data_receive)
    print('data_receive_hex: ', bytes2hex(data_receive))

    a, b, c = parse_one_frame(message=data_receive, total_length=7, codes='<f', head_length=2, tail_length=1)
    print('start_frame: ', a)
    print('data_frame: ', b)
    print('end_frame: ', c)

    message = build_message(head_frame='55 DA', data_frame=[2.23, 1.4, 34.3], tail_frame='B6', codes='<f')
    print('float message: ', message)
    a, b, c = parse_one_frame(message, total_length=15, codes='<f')
    print('parse message: ', a, b, c)


def multi_message_send_receive_test():
    x1, y1, z1 = 4.11, 5.77, 1.55
    x2, y2, z2 = 5.76, 1.33, 7.22
    check_sum = 0.0
    message1 = build_message(head_frame='55 DA', data_frame=[x1, y1, z1], tail_frame='FF AD', codes='<f')
    message2 = build_message(head_frame='55 DA', data_frame=[x2, y2, z2], tail_frame='FF AD', codes='<f')
    message3 = build_message(head_frame='55 D5', data_frame=[5.11], tail_frame='FF 5D', codes='<f')
    message4 = build_message(head_frame='55 D6', data_frame=[3.55], tail_frame='FF 6D', codes='<f')
    message5 = build_message(head_frame='55 D7', data_frame=[0.47], tail_frame='FF 7D', codes='<f')
    message = b''.join([message5, message3, message4, message2, message1])
    print('send_message: ', message)

    received_message = message
    # hex_message = received_message.hex()
    # print('hex_message: ', hex_message)
    # print(object2hex(3.44, codes='<f'))
    result = parse_multi_frame(received_message, codes='<f', head_length=2, tail_length=2)
    print(result)


def receive():
    """
    sudo chmod 777 /dev/ttyUSB1
    """
    # get_ports()
    ser = serial.Serial(port='/dev/ttyUSB0',
                        baudrate=115200,
                        bytesize=8,
                        stopbits=serial.STOPBITS_ONE,
                        parity=serial.PARITY_NONE,
                        rtscts=False,
                        timeout=None,
                        write_timeout=None)
    while True:
        if ser.inWaiting():
            data = []
            for i in range(ser.inWaiting()):
                s = ser.read(1).hex()
                # s = ser.read(1).decode('utf-8')
                data.append(s)
            print(data)


if __name__ == '__main__':
    # message_send_receive_test()
    multi_message_send_receive_test()
    # receive()
    # print(build_message(head_frame='ff', tail_frame='ee', data_frame=3.4, codes='<f'))
