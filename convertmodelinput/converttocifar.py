import gzip
import os
import re
import sys
import tarfile
import urllib
import string
import struct
#import glob
import random
import time
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_CHANNEL = 3
IMAGE_WIDTH   = 32
IMAGE_HEIGHT  = 32

#process image
def convertOneImageToCifar(labelId, imagePath, curfile):

    print imagePath
    #reszie pic , and convert it to rgb three channels
    image1 = Image.open(imagePath).resize((IMAGE_WIDTH, IMAGE_HEIGHT)).convert("RGB")

    imageArray = np.array(image1)

    imageReshapeArray = imageArray.reshape([IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])

    imageTranposeArray = np.transpose(imageReshapeArray, [2,0,1])

    imagedata = imageTranposeArray.reshape([IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNEL])

    fmt = "B" + str(IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNEL)+"s"

    buffer = struct.pack(fmt, np.uint8(labelId), imagedata.tostring())

    print "imagePath:", imagePath, "labelId:", labelId , " buffersize:", len(buffer)

    curfile.write(buffer)

def processImageToBatchBinFile(imagePaths, imageLabels, dstDir, batch):

    numPerBatch = int(len(imagePaths)/batch)

    batchIndex = 1

    num = 0

    curfile = None

    print "total image paths : ", len(imagePaths) , " num_per_batch:", numPerBatch

    for i in xrange(len(imagePaths)):

        imagePath  = imagePaths[i]

        labelId    = imageLabels[i]

        if (curfile == None):

            fileName = "data_batch_" + str(batchIndex) + ".bin"

            dstfile = os.path.join(dstDir, fileName)

            if(os.path.exists(dstfile)):
                os.remove(dstfile)
            curfile = open(dstfile, "wa")

        print imagePath

        convertOneImageToCifar(labelId, imagePath, curfile)


        print "process: ", batchIndex, "label: ", labelId ," num: ", num

        num = num + 1

        if (num == numPerBatch):
            # next batch
            batchIndex = batchIndex + 1
            num = 0
            curfile.close()
            curfile = None

    if (curfile != None):
        curfile.close()

def convertToCifar(imagePaths, imageLabels, dstDir, batch):

    processImageToBatchBinFile(imagePaths, imageLabels, dstDir, batch)






