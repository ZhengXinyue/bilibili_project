import time

import numpy as np
import cv2

import rospy
from sensor_msgs.msg import Image as Image_msg
from sensor_msgs.msg import PointCloud2

# path: /home/zhengxinyue/livox_ws/devel/lib/python2.7/dist-packages/livox_ros_driver/msg
from livox_ros._CustomMsg import CustomMsg

from sensor_msgs import point_cloud2
from ros_numpy.point_cloud2 import get_xyz_points
from ros_numpy.image import image_to_numpy, numpy_to_image

from livox_ros.yolox_demo import get_predictor

predictor = get_predictor()


def livox_callback(msg):
    pass
    # print('livox: ', msg.header.stamp.to_sec())
    # p = msg.points
    # start = time.time()
    # data = np.array([[i.x, i.y, i.z] for i in p])
    # print(time.time() - start)


def image_callback(msg):
    # print('image: ', msg.header.stamp.to_sec())
    image = image_to_numpy(msg)
    outputs, img_info = predictor.inference(image)
    result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
    image_publisher.publish(numpy_to_image(result_image, encoding='rgb8'))


if __name__ == '__main__':
    rospy.init_node('livox')
    livox_subscriber = rospy.Subscriber('/livox/lidar', CustomMsg, callback=livox_callback, queue_size=1)
    image_subscriber = rospy.Subscriber('/camera/color/image_raw', Image_msg, callback=image_callback, queue_size=1)
    image_publisher = rospy.Publisher('/detect_image', Image_msg, queue_size=1)
    rospy.spin()
