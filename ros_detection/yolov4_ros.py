import sys

import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image
from ros_numpy.image import image_to_numpy, numpy_to_image


def draw_bboxes(img, bboxes, color=(0, 0, 255)):
    for bbox in bboxes:
        bbox = np.array(bbox, dtype=np.int32)
        left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
        cv2.rectangle(img, (left, top), (right, bottom), color, int(max(img.shape[:2]) / 200))
    return img


class Yolov4Ros(object):
    def __init__(self, predictor):
        self.predictor = predictor
        self.image_subscriber = rospy.Subscriber('/tracking_image', Image, callback=self.image_callback, queue_size=1)
        self.image_publisher = rospy.Publisher('/image_publish', Image, queue_size=1)

    def image_callback(self, msg):
        image = image_to_numpy(msg)

        result_image = image
        self.image_publisher.publish(numpy_to_image(result_image, encoding='bgr8'))


if __name__ == '__main__':
    # https://github.com/Tianxiaomo/pytorch-YOLOv4
    detector = None
    rospy.init_node('yolov4_ros')
    yolov4_ros = Yolov4Ros(predictor=detector)
    rospy.spin()
