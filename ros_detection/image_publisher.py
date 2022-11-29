import os

import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image

from ros_numpy.image import numpy_to_image


if __name__ == '__main__':
    image_dir = '../data_example/detection/images'
    image_list = os.listdir(image_dir)
    image_list = [os.path.join(image_dir, i) for i in image_list]
    image_iter = iter(image_list)

    rospy.init_node('image_publish')
    publisher = rospy.Publisher('/test_image', Image, queue_size=1)
    loop_rate = rospy.Rate(1)
    while True:
        try:
            image_path = next(image_iter)
        except StopIteration:
            image_iter = iter(image_list)
            image_path = next(image_iter)
        image = cv2.imread(image_path)
        image_msg = numpy_to_image(image, encoding='bgr8')
        publisher.publish(image_msg)
        loop_rate.sleep()
