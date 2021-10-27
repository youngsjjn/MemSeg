import os
import os.path
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

rellis_label_mapping = {
            1: 0, 3: 1,  4: 2, 5: 3,
            6: 4, 7: 5,
            8: 6, 9: 7, 10: 8, 12: 9, 15: 10, 17: 11,
            18: 12, 19: 13, 23: 14, 27: 15,
            31: 16, 33: 17, 34: 18}

def rellis_labelID2trainID(img):
    temp = np.copy(img)
    for k, v in rellis_label_mapping.items():
        temp[img == k] = v
    return temp

def label_to_long_tensor(pic):
    label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    label = label.view(pic.size[1], pic.size[0], 1)
    label = label.transpose(0, 1).transpose(0, 2).squeeze().contiguous().long()
    return label

def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            if 'CamVid' in data_list:
                image_name = data_root + line_split[0]
                # label_name = data_root + line_split[0].replace(split, split + '_Glabels').replace('.png','_L.png')
                label_name = data_root + line_split[1]
            else:
                image_name = os.path.join(data_root, line_split[0])
                label_name = os.path.join(data_root, line_split[1])
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # image_path = '/home/ispl3/Documents/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000001_014565_leftImg8bit.png'
        # label_path = '/home/ispl3/Documents/cityscapes/gtFine/val/munster/munster_000150_000019_gtFine_labelTrainIds.png'
        image_path, label_path = self.data_list[index]
        # image_path = '/home/ispl3/PycharmProjects/pytorch/MemSeg/Wild_image/park-1_02671.png'
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        if 'CamVid' in self.data_list[0][0]:
            label[label == 11] = 255
        elif 'RELLIS' in self.data_list[0][0]:
            label[label == 0] = 255
            label = rellis_labelID2trainID(label)


        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label
