import os

import cv2
import numpy as np

import rospy
from std_msgs.msg import String, Header, Int8
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, Image as Image_msg
from sensor_msgs import point_cloud2 as pc2
from ros_numpy.point_cloud2 import array_to_pointcloud2
from ros_numpy.image import numpy_to_image


def numpy2cloud_array(array, target_fields=None):
    """
    转换点云需要的fields
    :param array:            numpy.ndarray   `N x len(target_fields)`
    :param target_fields:    ((field_name, field_dtype), ...)
                              [('x', np.float32), ('y', np.float32), ('z', np.float32),
                               ('r', np.uint8), ('g', np.uint8), ('b', np.uint8),
                               ('intensity', np.uint32)]
    :return:  new_array: numpy.ndarray
    """
    if target_fields is None:
        target_fields = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
    new_array = np.zeros(shape=(array.shape[0]), dtype=target_fields)
    for i, target_field in enumerate(target_fields):
        field_name = target_field[0]
        new_array[field_name] = array[:, i]
    return new_array


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


def numpy2cloud_msg(array, target_fields=None, stamp=None, frame_id=None):
    """
    转换点云需要的fields
    :param frame_id:         /map   /os_lidar ...
    :param stamp:            rospy.Time.now() ...
    :param array:            numpy.ndarray   `N x len(target_fields)`
    :param target_fields:    ((field_name, field_dtype), ...)
                              [('x', np.float32), ('y', np.float32), ('z', np.float32),
                               ('r', np.uint8), ('g', np.uint8), ('b', np.uint8),
                               ('intensity', np.uint32)]
    :return:  cloud_msg: sensor_msgs.msg.PointCloud2
    >>> point_cloud = read_bin('data_example/3d_detection/velodyne/000006.bin', intensity=True)
    >>> pointcloud_publisher = rospy.Publisher('/test_pointcloud', PointCloud2, queue_size=1)
    >>> pointcloud_publisher.publish(numpy2cloud_msg(point_cloud,
    >>>                                              target_fields=[('x', np.float32), ('y', np.float32),
    >>>                                                             ('z', np.float32), ('intensity', np.float32)],
    >>>                                              stamp=rospy.Time.now(),
    >>>                                              frame_id='/map'))
    """
    new_array = numpy2cloud_array(array, target_fields=target_fields)
    cloud_msg = array_to_pointcloud2(new_array, stamp=stamp, frame_id=frame_id)
    return cloud_msg


if __name__ == '__main__':
    rospy.init_node('publisher_example')
    frequency = 5
    loop_rate = rospy.Rate(frequency, reset=True)
    image_publisher = rospy.Publisher('/tracking_image', Image_msg, queue_size=1)
    a = list(os.walk('../data_example/detection/images'))[0]
    paths = [os.path.join(a[0], i) for i in a[2]]
    images = [cv2.imread(i) for i in paths]
    image_iter = iter(images)

    pointcloud_publisher = rospy.Publisher('/test_pointcloud', PointCloud2, queue_size=1)
    b = list(os.walk('../data_example/3d_detection/velodyne'))[0]
    paths = [os.path.join(b[0], i) for i in b[2]]
    points = [read_bin(i, intensity=True) for i in paths]
    point_iter = iter(points)

    while not rospy.is_shutdown():
        try:
            image = next(image_iter)
        except StopIteration:
            image_iter = iter(images)
            image = next(image_iter)
        image_publisher.publish(numpy_to_image(image, encoding='bgr8'))

        try:
            point = next(point_iter)
        except StopIteration:
            point_iter = iter(points)
            point = next(point_iter)
        pointcloud_publisher.publish(numpy2cloud_msg(point, frame_id='/map', target_fields=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)]))

        loop_rate.sleep()
