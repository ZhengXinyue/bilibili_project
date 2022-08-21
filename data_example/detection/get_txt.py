import os

# 生成train.txt
xml_dir = 'annotations'
img_dir = 'images'
path_list = list()
for img in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img)  # 获取图像路径
    xml_path = os.path.join(xml_dir, img.replace('jpg', 'xml'))  # 获取图像对应的标注文件的路径
    path_list.append((img_path, xml_path))

train_f = open('trainval.txt', 'w')

# 用一个迭代器将"图片路径 标签路径"保存到.txt文件中
for i, (img_path, xml_path) in enumerate(path_list):
    text = img_path + " " + xml_path + "\n"
    train_f.write(text)

train_f.close()

# 生成标签文档
label = ['car', 'horse', 'person', 'cat']

with open('label_list.txt', 'w') as f:
    for text in label:
        f.write(text + '\n')
