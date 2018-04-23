import shutil
import os
import random
import pandas as pd


# import and use in python console
# split_data('E:/work/data/driver_detection/train', 'E:/work/data/driver_detection/validation', 0.2)
# this function do shuffle during splitting
def split_data(src, des, rate):
    if len(os.listdir(des)) != 0:
        print('{0} is not empty'.format(des))
        return

    classes = os.listdir(src)
    print('classes num: {0}'.format(len(classes)))
    print(classes)

    for cls in classes:
        src_cls_path = os.path.join(src, cls)
        cls_imgs = os.listdir(src_cls_path)
        src_count = len(cls_imgs)
        des_count = int(src_count*rate)
        print('----------------------------------')
        print('{0} img num: {1}'.format(cls, src_count))
        des_cls_path = os.path.join(des, cls)

        os.mkdir(des_cls_path)

        move_count = 0
        progress = list(range(10, 0, -1))
        for img in cls_imgs:
            ra = random.random()
            if ra <= rate:
                src_img_path = os.path.join(src_cls_path, img)
                des_img_path = os.path.join(des_cls_path, img)
                shutil.move(src_img_path, des_img_path)
                move_count += 1
                if move_count == des_count:
                    print('{0} finish'.format(cls))
                    break
                if progress[len(progress)-1] < move_count/des_count*10:
                    print('progress: {0}%'.format(move_count/des_count*100))
                    progress.pop()
        print('real move num: {0}'.format(move_count))
        print('need to move: {0}'.format(des_count))


# 这个函数用来把糖网图片按照trainLables.csv的标注分放到不同目录
# split_dia_data('E:/work/data/Diabetic_Retinopathy_Detection/sample',
#                'E:/work/data/Diabetic_Retinopathy_Detection/all',
#                'E:/work/data/Diabetic_Retinopathy_Detection/trainLabels.csv')
def split_dia_data(src, des, lable_path):
    # 创建分类目录
    classes = ['0', '1', '2', '3', '4']
    for cls in classes:
        os.mkdir(os.path.join(des, cls))

    lables = pd.read_csv(lable_path)
    for row in lables.values:
        img_name = row[0]+'.jpeg'
        cls = str(row[1])
        src_img_path = os.path.join(src, img_name)
        if os.path.exists(src_img_path):
            des_img_path = os.path.join(des, cls, img_name)
            shutil.move(src_img_path, des_img_path)