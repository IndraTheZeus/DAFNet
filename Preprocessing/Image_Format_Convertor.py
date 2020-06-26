# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 01:03:00 2019

@author: indra
"""

from PIL import Image
import glob
def convert_image_to_type(filename,convert_type,destination):
    img = Image.open(filename)
    filename = filename[filename.rfind('\\')+1:]
    img.save(destination+filename[:-3]+convert_type,convert_type)
    
    
image_glob = 'D:\\STANDARD DATASETS\\ICDAR 2017\\groundtruth_pixel\\Train\\ToOverfit\\*.jpg'
destination = 'D:\\STANDARD DATASETS\\ICDAR 2017\\groundtruth_pixel\\Train\\png\\'


filenames = glob.glob(image_glob)
print(filenames)
for filename in filenames:
    convert_image_to_type(filename,'png',destination)
