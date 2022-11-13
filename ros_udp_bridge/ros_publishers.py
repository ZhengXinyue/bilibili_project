import numpy as np

import rospy

from std_msgs.msg import Float32MultiArray, UInt8
from sensor_msgs.msg import PointCloud2

from ros_numpy.point_cloud2 import array_to_pointcloud2


def read_bin(bin_path, intensity=False):
    """
    读取kitti bin格式文件点云
    :param bin_path:   点云路径
    :param intensity:  是否要强度
    :return:           numpy.ndarray `N x 3` or `N x 4`
    """
    lidar_points = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
    if not intensity:
        lidar_points = lidar_points[:, :3]
    return lidar_points


def numpy2cloud_array(array, target_fields=None):
    """
    转换点云需要的fields
    :param array:            numpy.ndarray   `N x len(target_fields)`
    :param target_fields:    ((field_name, field_dtype), ...)
                              [('x', np.float32), ('y', np.float32), ('z', np.float32),
                               ('r', np.float32), ('g', np.float32), ('b', np.float32),
                               ('intensity', np.float32)]
    :return:  new_array: numpy.ndarray
    """
    if target_fields is None:
        target_fields = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
    new_array = np.zeros(shape=(array.shape[0]), dtype=target_fields)
    for i, target_field in enumerate(target_fields):
        field_name = target_field[0]
        new_array[field_name] = array[:, i]
    return new_array


def numpy2cloud_msg4(array, frame_id, target_fields=None, stamp=None):
    """
    :param frame_id:         /map
    :param stamp:            rospy.Time.now() ...
    :param array:            numpy.ndarray   `N x len(target_fields)`
    :param target_fields:    ((field_name, field_dtype), ...)
                              [('x', np.float32), ('y', np.float32), ('z', np.float32),
                               ('r', np.uint8), ('g', np.uint8), ('b', np.uint8),
                               ('intensity', np.float32)]     r, g, b 范围为[0, 255]
    :return:  cloud_msg: sensor_msgs.msg.PointCloud2
    """
    new_array = numpy2cloud_array(array, target_fields=target_fields)
    cloud_msg = array_to_pointcloud2(new_array, stamp=stamp, frame_id=frame_id)
    return cloud_msg


if __name__ == '__main__':
    pc2_points = read_bin('../data_example/3d_detection/velodyne/000003.bin', intensity=False)   # N x 3

    rospy.init_node('ros_publishers')
    pub_frequency = 0.5
    loop_rate = rospy.Rate(pub_frequency, reset=True)

    pc2_publisher = rospy.Publisher('/pc2_topic', PointCloud2, queue_size=1)
    uint8_publisher = rospy.Publisher('/uint8_topic', UInt8, queue_size=1)
    float32_ma_publisher = rospy.Publisher('/float32_ma_topic', Float32MultiArray, queue_size=1)

    uint8_data = 0
    float32_ma_data = np.array([1, 2, 3], dtype=np.float32)   # [age, weight, height]

    while True:
        # generate message
        uint8_msg = UInt8(data=uint8_data)
        float32_ma_msg = Float32MultiArray(data=float32_ma_data.tolist())
        pc2_msg = numpy2cloud_msg4(pc2_points, frame_id='/map')

        # publish message
        uint8_publisher.publish(uint8_msg)
        float32_ma_publisher.publish(float32_ma_msg)
        pc2_publisher.publish(pc2_msg)

        uint8_data += 1
        float32_ma_data += 1

        loop_rate.sleep()
