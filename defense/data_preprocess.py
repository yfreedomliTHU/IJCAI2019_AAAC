# DATA preprocess for AAAC
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import PIL
from PIL import Image
from io import BytesIO
from piexif._common import *
from piexif import _webp



def exif_remove(src):
    if src[0:2] == b"\xff\xd8":
        src_data = src
        file_type = "jpeg"
    elif src[0:4] == b"RIFF" and src[8:12] == b"WEBP":
        src_data = src
        file_type = "webp"
    else:
        return src
    new_data = src_data
    if file_type == "jpeg":
        segments = split_into_segments(src_data)
        exif = get_exif_seg(segments)
        if exif:
            new_data = src_data.replace(exif, b"")
        else:
            new_data = src_data
    elif file_type == "webp":
        try:
            new_data = _webp.remove(src_data)
        except:
            new_data = src_data
    return new_data



def process(input_dir, output_dir):

    file_name = os.listdir(input_dir)

    for filename in file_name:
        file_path = os.path.join(input_dir, filename)

        with open(file_path, 'rb') as f:
            image_bytes = f.read()
            image_bytes = exif_remove(image_bytes)
        iof = BytesIO()
        iof.write(image_bytes)
        image = Image.open(iof)
        x, y = image.size
        if x < 30 or y < 30:
            print(filename)
            continue
        if image.format == "GIF":
            continue
        if image.mode == 'RGBA':
            r, g, b, a = image.split()
            image = Image.merge("RGB", (r, g, b))
        if image.mode != "RGB":
            continue
        image = image.resize(img_size, Image.BILINEAR)
        image.save(os.path.join(output_dir, filename), format='JPEG')
        iof.close()
        f.close()


if __name__ == '__main__':
    data_path = './IJCAI_2019_AAAC_train_data/IJCAI_2019_AAAC_train'
    save_path = 'IJCAI_2019_AAAC_train_procrssed'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    Class_list = os.listdir(data_path)
    img_size = (299, 299)
    for Class in Class_list:
        Input_dir = os.path.join(data_path, Class)
        Output_dir = os.path.join(save_path, Class)
        if not os.path.exists(Output_dir):
            os.mkdir(Output_dir)
        process(Input_dir, Output_dir)