import sys
CENTERNET_PATH = '/path/to/CenterNet/src/lib/'
sys.path.insert(0, CENTERNET_PATH)
from detectors.detector_factory import detector_factory
from opts import opts

import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image
from ros_numpy.image import image_to_numpy, numpy_to_image


def draw_bboxes(img, info, color=(0, 0, 255)):
    for k, bboxes in info.items():
        for bbox in bboxes:
            bbox = np.array(bbox, dtype=np.int32)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            cv2.rectangle(img, (left, top), (right, bottom), color, int(max(img.shape[:2]) / 200))
    return img


class CenterNetRos(object):
    def __init__(self, predictor):
        self.predictor = predictor
        self.image_subscriber = rospy.Subscriber('/tracking_image', Image, callback=self.image_callback, queue_size=1)
        self.image_publisher = rospy.Publisher('/image_publish', Image, queue_size=1)

    def image_callback(self, msg):
        image = image_to_numpy(msg)

        ret = detector.run(image)['results']
        # ret will be a python dict: {category_id : [[x1, y1, x2, y2, score], ...], }
        # ret = {3: [[3, 4, 50, 70], [44, 44, 99, 99]]}
        result_image = draw_bboxes(image, ret)
        self.image_publisher.publish(numpy_to_image(result_image, encoding='bgr8'))


if __name__ == '__main__':
    # https://github.com/xingyizhou/CenterNet
    MODEL_PATH = '/ path / to / model'
    TASK = 'ctdet'  # or 'multi_pose' for human pose estimation
    opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
    detector = detector_factory[opt.task](opt)

    # detector = None
    rospy.init_node('centernet_ros')
    centernet_ros = CenterNetRos(predictor=detector)
    rospy.spin()
