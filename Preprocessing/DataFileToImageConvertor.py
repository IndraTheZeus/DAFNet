# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 21:57:30 2020

@author: indra
"""

from scipy.io import loadmat
from os import listdir
from PIL import Image,ImageDraw
matFilesDir = 'D:\\STANDARD DATASETS\\ICDAR 2017\\groundtruth_text\\Groundtruth\\Polygon\\Test'
imageFilesDir = 'D:\\STANDARD DATASETS\\ICDAR 2017\\totaltext\\Images\\Test'
outputDir = 'D:\\STANDARD DATASETS\\ICDAR 2017\\MyTestGT3'

def drawPolygon(DrawImage,coords):
    DrawImage.polygon(coords, fill =(255),outline=(0)) 
    
def drawFalsePolygon(DrawImage,coords):
    DrawImage.polygon(coords, fill =(63),outline=(0))
    
   
    
def drawBoundingBoxes_ICDAR2017():
     fileList = listdir(matFilesDir)
     print(fileList)
     for filename in fileList:
         try:
           mat = loadmat(matFilesDir +'\\'+filename)
           image_size = Image.open(imageFilesDir+'\\'+filename[8:-3]+'jpg').size
           bb_img = Image.new("L", image_size, color=(0))  
           draw_img = ImageDraw.Draw(bb_img) 
           for word_desc in mat['polygt']:
                assert word_desc[0][0] == 'x:'
                xcoords = word_desc[1][0]
                assert word_desc[2][0] == 'y:'
                ycoords = word_desc[3][0]
                assert xcoords.size == ycoords.size
                if((xcoords.size < 2) & (word_desc[4][0] != '#')):
                   print(word_desc)
                   assert False
                if(xcoords.size <2):
                 continue
                xy = []
                for i in range(xcoords.size):
                     xy.append(xcoords[i])
                     xy.append(ycoords[i])
                if(word_desc[4][0] == '#'):
                   drawFalsePolygon(draw_img,xy)
                else:
                   drawPolygon(draw_img,xy)
           bb_img.save(outputDir+'\\'+filename[8:-3]+'png',"PNG")
         except:
            print(filename)

             
drawBoundingBoxes_ICDAR2017()