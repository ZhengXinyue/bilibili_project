import numpy as np
import struct

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from ros_numpy.point_cloud2 import array_to_pointcloud2, merge_rgb_fields


def numpy2cloud_msg1(points, colors, frame_id, stamp=None, seq=None):
    """
    @param points:      numpy.ndarray float32
    @param colors:      numpy.ndarray uint8  [0, 255]

    """
    msg = PointCloud2()
    if stamp is not None:
        msg.header.stamp = stamp
    if frame_id is not None:
        msg.header.frame_id = frame_id
    if seq is not None:
        msg.header.seq = seq

    colors = (colors / 255).astype(np.float32)
    data = np.hstack((points, colors))   # (N, 3) + (N, 3) --> (N, 6)

    msg.height = 1
    msg.width = len(points)
    msg.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('r', 12, PointField.FLOAT32, 1),
                  PointField('g', 16, PointField.FLOAT32, 1),
                  PointField('b', 20, PointField.FLOAT32, 1)]
    msg.is_bigendian = False
    msg.point_step = 24
    msg.row_step = msg.point_step * len(points)
    msg.is_dense = np.isfinite(points).all()
    msg.data = data.tobytes()

    return msg


def numpy2cloud_msg2(points, colors, frame_id, stamp=None, seq=None):
    """
    https://github.com/ros-visualization/rviz/pull/761
    https://github.com/ros-visualization/rviz/blob/1.9.36/src/rviz/default_plugin/point_cloud_transformers.cpp#L329
    发布带颜色的点云
    @param points:      numpy.ndarray float32  N x 3 (x, y, z)
    @param colors:      numpy.ndarray uint8    N x 3 (r, g, b)    r, g, b 范围为 [0, 255]
    @param frame_id:
    @param stamp:
    @param seq:
    @return:
    """
    r, g, b = colors[:, 0].astype(np.uint32), colors[:, 1].astype(np.uint32), colors[:, 2].astype(np.uint32)
    rgb = np.array((r << 16) | (g << 8) | (b << 0), dtype=np.uint32)
    rgb = np.reshape(rgb, newshape=(rgb.shape[0], 1))
    rgb.dtype = np.float32
    array = np.hstack((points, rgb))

    msg = PointCloud2()
    if stamp is not None:
        msg.header.stamp = stamp
    if frame_id is not None:
        msg.header.frame_id = frame_id
    if seq is not None:
        msg.header.seq = seq
    msg.height = 1
    msg.width = len(array)
    msg.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('rgb', 12, PointField.FLOAT32, 1)]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * array.shape[0]
    msg.is_dense = np.isfinite(array).all()
    msg.data = array.tobytes()
    return msg


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


def numpy2cloud_msg3(array, frame_id, target_fields=None, stamp=None):
    """
    使用ros_numpy, 本质代码一致
    转换点云需要的fields
    :param frame_id:         /map
    :param stamp:            rospy.Time.now() ...
    :param array:            numpy.ndarray   `N x len(target_fields)`
    :param target_fields:    ((field_name, field_dtype), ...)
                              [('x', np.float32), ('y', np.float32), ('z', np.float32),
                               ('r', np.float32), ('g', np.float32), ('b', np.float32),
                               ('intensity', np.float32)]     r, g, b 范围为[0, 1]
    :return:  cloud_msg: sensor_msgs.msg.PointCloud2
    """
    new_array = numpy2cloud_array(array, target_fields=target_fields)
    cloud_msg = array_to_pointcloud2(new_array, stamp=stamp, frame_id=frame_id)
    return cloud_msg


def numpy2cloud_msg4(array, frame_id, target_fields=None, stamp=None):
    """
    使用ros_numpy并压缩rgb
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
    new_array = merge_rgb_fields(new_array)
    cloud_msg = array_to_pointcloud2(new_array, stamp=stamp, frame_id=frame_id)
    return cloud_msg


if __name__ == '__main__':
    points = np.load('../data_example/points.npy')   # np.float32
    colors = np.load('../data_example/colors.npy')   # np.uint8

    rospy.init_node('colored_pc2_publish')
    pc2_publisher = rospy.Publisher('/test_pointcloud', PointCloud2, queue_size=1)
    loop_rate = rospy.Rate(5, reset=True)

    while not rospy.is_shutdown():
        pc2_msg = numpy2cloud_msg1(points, colors, '/map')
        # pc2_msg = numpy2cloud_msg2(points, colors, '/map')

        # 使用ros_numpy
        # pc2_msg = numpy2cloud_msg3(np.hstack((points, colors/255)), '/map',
        #                            target_fields=[('x', np.float32), ('y', np.float32), ('z', np.float32),
        #                                           ('r', np.float32), ('g', np.float32), ('b', np.float32)])
        # pc2_msg = numpy2cloud_msg4(np.hstack((points, colors)), '/map',
        #                            target_fields=[('x', np.float32), ('y', np.float32), ('z', np.float32),
        #                                           ('r', np.uint8), ('g', np.uint8), ('b', np.uint8)])
        pc2_publisher.publish(pc2_msg)
        loop_rate.sleep()







