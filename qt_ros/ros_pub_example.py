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
from geometry_msgs.msg import PoseWithCovariance, TwistWithCovariance, Pose, Point, Quaternion, Twist, Vector3
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

from utils.file_op import read_bin
from utils.transform import numpy2cloud_msg


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
