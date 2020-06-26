# -*- coding: utf-8 -*-
"""
Created on Fri May 22 18:38:48 2020

@author: indra
"""

from PIL import Image, ImageEnhance
from os import listdir

orig_dir = 'D:\\STANDARD DATASETS\\ICDAR 2017\\totaltext\\Images\\Test'
enhance_dir = 'D:\\STANDARD DATASETS\\ICDAR 2017\\totaltext\\Images\\Test-enhanced'


fileList = listdir(orig_dir)
for file in fileList:
    if file[-4:] == '.jpg':
       #read the image
       im = Image.open(orig_dir+'\\'+file)

        #image brightness enhancer
       enhancer = ImageEnhance.Contrast(im)

       factor = 2 #increase contrast
       im_output = enhancer.enhance(factor)
       im_output.save(enhance_dir+'\\'+file[:-3]+'png')        


