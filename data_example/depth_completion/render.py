from utils.image_utils import get_colored_depth
from utils.file_op import read_depth
import cv2
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def f():
    sparse_depth = read_depth('data_example/depth_completion/data_depth_velodyne/0000000005.png')
    colored_sparse_depth = get_colored_depth(sparse_depth)
    semi_dense_depth = read_depth('data_example/depth_completion/data_depth_annotated/0000000005.png')
    colored_semi_dense_depth = get_colored_depth(semi_dense_depth)
    cv2.imwrite('colored_semi_dense.png', colored_semi_dense_depth)
    cv2.imwrite('colored_sparse.png', colored_sparse_depth)


if __name__ == '__main__':
    f()
