import gzip
import os
import re
import sys
import tarfile
import urllib
import tensorflow as tf
import string
import numpy as np
import PIL as Image
from StringIO import StringIO

"""
 a script used to preprocess images
"""
from util import Option

#parse env path

G_OPTION = Option()

#read file parse lable index and label name
def parseLabelPath(labelPath):

    dict={}

    file = open(labelPath, "rb")

    index = 0;

    for line in file:
        line = line.strip("\n")
        dict[index] = line;
        index=index+1

#get image index
def getImageIndex(indexFile):

    imageIndexFile = open(indexFile, 'rb')

    imageDict={}
    for line in imageIndexFile:
        line = line.strip("\n")

        item = line.split("|")

        if(len(item) == 2):
            imageName = item[0];
            imageLabel= string.atoi(item[1])
            imageDict[imageName] = imageLabel
        else:
            print " image file error: ",line

    return imageDict


def subtractmean():
    return


def normalize():
    return


def whitening():
    return


def crop():
    return


def brightness():
    return


def contrast():
    return

#process image
def processOneImage(imageFile, imageLabel):

    image  = Image.open(imageFile, "rb");

    buffer = np.array(image);

    IMAGE_WIDTH   = string.atoi(G_OPTION.getValue("width"));
    IMAGE_HEIGHT  = string.atoi(G_OPTION.getValue("height"));
    IMAGE_CHANNEL = string.atoi(G_OPTION.getValue("channel"));

    buffer = buffer.reshape([IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])

    mode = G_OPTION.getValue("mode")

    for char in mode:
        if(char == 'm'):
            #substract
            buffer=subtractmean(buffer)
            return
        else:
            #normalize
            if(char == 'n'):
                return
            else:
                #whitening
                if(char == "w"):
                    return
                else:
                    if(char == "c"):
                        return



#preprocess image
def preProcess(labelPath, imageDir):

    labelDict = parseLabelPath(labelPath)

    indexFile = imageDir + "/index.meta";

    imageDict = getImageIndex(indexFile)

    if(len(imageDict) == 0):
        print "image index error "
        exit(-1)

    for imageName, imageLabel in imageDict:
        imageFilePath = imageDir + "/" + imageName;
        processOneImage(imageFilePath, imageLabel)






def useage():
    print "./preprocess labelpath imagedir mode"
    print  "labelpath type: str, represent label path"
    print  "imagedir  type: str, represent image dir"
    print  "mode      type: str, represent defferent process type:"
    print  "          m: subtract mean"
    print  "          n: normlize"
    print  "          w: whitening"
    print  "          c: crop"
    print  "          f: flip image"
    print  "          b: brightness"
    print  "          c: contrast "



def getEnv():

    if(len(sys.argv) < 4):
        useage()
        exit(-1)

    print __name__ + "start------------------"

    G_OPTION.decode(sys.argv)

    labelPath = G_OPTION.getValue("labelPath");

    imageDir  = G_OPTION.getValue("imagedir");

    print "labelPath",labelPath

    print "imagedir",imageDir

    print __name__ + "end------------------"

    return labelPath, imageDir;


def main():
    labelPath, imageDir= getEnv()
    preProcess(labelPath, imageDir)


if __name__ == "__main__":
    main()