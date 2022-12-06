import time
import os
import sys
sys.path.append('/home/zhengxinyue/YOLOX')

import torch
import cv2
import numpy as np

from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.data.data_augment import ValTransform


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        fp16=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=False)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, image):
        img_info = {}
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.fp16:
            img = img.half()  # to FP16

        # with torch.no_grad():
        outputs = self.model(img)
        if self.decoder is not None:
            outputs = self.decoder(outputs, dtype=outputs.type())
        outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre, class_agnostic=True)
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def get_predictor():
    exp_name = None
    model_name = 'yolox-nano'
    ckpt_file = '/home/zhengxinyue/YOLOX/yolox_nano.pth'
    conf = 0.4
    nms = 0.3
    test_size = (416, 416)
    fp16 = False
    fuse = False
    trt_file = None

    exp = get_exp(exp_name, model_name)
    exp.test_conf = conf
    exp.nmsthre = nms
    exp.test_size = test_size
    print(exp)

    model = exp.get_model()

    if fp16:
        model.half()
    torch.set_grad_enabled(False)
    model.eval()

    if trt_file is None:
        ckpt = torch.load(ckpt_file, map_location='cpu')
        model.load_state_dict(ckpt['model'])

    if fuse:
        model = fuse_model(model)

    if trt_file is None:
        decoder = None
    else:
        assert not fuse, "TensorRT model is not support model fusing!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs

    predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, fp16)
    return predictor


if __name__ == '__main__':
    exp_name = None
    model_name = 'yolox-nano'
    ckpt_file = '/home/zhengxinyue/YOLOX/yolox_nano.pth'
    conf = 0.4
    nms = 0.3
    test_size = (416, 416)
    fp16 = False
    fuse = False
    trt_file = None

    exp = get_exp(exp_name, model_name)
    exp.test_conf = conf
    exp.nmsthre = nms
    exp.test_size = test_size
    print(exp)

    model = exp.get_model()
    model.cuda()

    if fp16:
        model.half()
    torch.set_grad_enabled(False)
    model.eval()

    if trt_file is None:
        ckpt = torch.load(ckpt_file, map_location='cpu')
        model.load_state_dict(ckpt['model'])

    if fuse:
        model = fuse_model(model)

    if trt_file is None:
        decoder = None
    else:
        assert not fuse, "TensorRT model is not support model fusing!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs

    predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, fp16)

    outputs, img_info = predictor.inference('../data_example/detection/images/000007.jpg')
    print(outputs[0].cpu())
    result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
    cv2.imwrite('result.jpg', result_image)
