"""
把coco数据集合的所有标注转换到voc格式，不改变图片命名方式，
注意，原来有一些图片是黑白照片，检测出不是 RGB 图像，这样的图像不会被放到新的文件夹中
代码源自：https://blog.csdn.net/qq_44955984/article/details/124612645
"""
import cv2
import os
import shutil

from PIL import Image
from lxml import etree, objectify
from pycocotools.coco import COCO
from tqdm import tqdm

# 1、生成图片保存的路径
CKimg_dir = 'data_voc/images'

# 2、生成标注文件保存的路径
CKanno_dir = 'data_voc/annotations'


# 若模型保存文件夹不存在，创建模型保存文件夹，若存在，删除重建
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)


def save_annotations(filename, objs, filepath):
    annopath = os.path.join(anno_dir, filename[:-3] + 'xml')  # 生成的xml文件保存路径
    dst_path = os.path.join(image_dir, filename)
    img_path = filepath
    img = cv2.imread(img_path)
    im = Image.open(img_path)

    im.close()
    shutil.copy(img_path, dst_path)  # 把原始图像复制到目标文件夹
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('1'),
        E.filename(filename),
        E.source(
            E.database('CKdemo'),
            E.annotation('VOC'),
            E.image('CK')
        ),
        E.size(
            E.width(img.shape[1]),
            E.height(img.shape[0]),
            E.depth(img.shape[2])
        ),
        E.segmented(0)
    )
    for obj in objs:
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name(obj[0]),
            E.pose(),
            E.truncated("0"),
            E.difficult(0),
            E.bndbox(
                E.xmin(obj[2]),
                E.ymin(obj[3]),
                E.xmax(obj[4]),
                E.ymax(obj[5])
            )
        )
        anno_tree.append(anno_tree2)
    etree.ElementTree(anno_tree).write(annopath, pretty_print=True)


def showbycv(coco, img, classes, origin_image_dir, verbose=False):
    filename = img['file_name']
    filepath = os.path.join(origin_image_dir, filename)
    I = cv2.imread(filepath)
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    objs = []
    for ann in anns:
        name = classes[ann['category_id']]
        if 'bbox' in ann:
            bbox = ann['bbox']
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2] + bbox[0])
            ymax = int(bbox[3] + bbox[1])
            obj = [name, 1.0, xmin, ymin, xmax, ymax]
            objs.append(obj)
            if verbose:
                cv2.rectangle(I, (xmin, ymin), (xmax, ymax), (255, 0, 0))
                cv2.putText(I, name, (xmin, ymin), 3, 1, (0, 0, 255))
    save_annotations(filename, objs, filepath)
    if verbose:
        cv2.imshow("img", I)
        cv2.waitKey(0)


def catid2name(coco):  # 将名字和id号建立一个字典
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
        # print(str(cat['id'])+":"+cat['name'])
    return classes


def get_CK5(annpath, origin_image_dir, verbose=False):
    coco = COCO(annpath)
    classes = catid2name(coco)
    imgIds = coco.getImgIds()
    # imgIds=imgIds[0:1000]#测试用，抽取10张图片，看下存储效果
    for imgId in tqdm(imgIds):
        img = coco.loadImgs(imgId)[0]
        # showbycv(coco, dataType, img, classes, origin_image_dir, verbose=False)
        showbycv(coco, img, classes, origin_image_dir, verbose=False)


if __name__ == "__main__":
    base_dir = 'data_voc'  # step1 这里是一个新的文件夹，存放转换后的图片和标注
    image_dir = os.path.join(base_dir, 'images')  # 在上述文件夹中生成images，annotations两个子文件夹
    anno_dir = os.path.join(base_dir, 'annotations')
    mkr(base_dir)
    mkr(image_dir)
    mkr(anno_dir)
    origin_image_dir = 'images'  # step 2原始的coco的图像存放位置
    anno_path = 'voc_train.json'  # step 3 原始的coco的标注存放位置
    print(anno_path)
    verbose = True  # 是否需要看下标记是否正确的开关标记，若是true,就会把标记展示到图片上
    get_CK5(anno_path, origin_image_dir, verbose)
