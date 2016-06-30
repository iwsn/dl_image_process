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
import shutil


from util.Option import *
from config import *
from dataset.datasetgenerator import *

"""
    a scrpit used to split our data set into train and validation data set
"""

G_OPTION = Option()

def getLabelList(srcDir):

    labelListFile = os.path.join(srcDir, LABEL_FILE_NAME);

    if(not os.path.exists(labelListFile)):
        print(" src dir does not exist label.txt file")
        exit(-1)

    labelFile = open(labelListFile, 'r')

    index = 0

    mapDict = {}

    for line in labelFile:
        line = line.strip("\n")
        if(line == ""):
            continue

        items = line.split("|")

        if(len(items) < 2):
            print("label list file error")
            exit(-1)

        synxnet   = items[0]
        humanName = items[1]

        labelRecord = LabelRecord(index, synxnet, humanName)

        mapDict[index] = labelRecord;

        index = index + 1

    return mapDict

def copyImageFile(srcDir, mapLabels, dstDir, imageList):
     if os.path.exists(dstDir):
         print("delete image dir ",dstDir)
         shutil.rmtree(dstDir)

     os.mkdir(dstDir)

     #copy label file
     labelSrcFile = os.path.join(srcDir, LABEL_FILE_NAME)
     labelDstFile = os.path.join(dstDir, LABEL_FILE_NAME)

     shutil.copyfile(labelSrcFile, labelDstFile)

     #create label dir
     for labelId in mapLabels:
         labelRecord = mapLabels[labelId]
         synxnet = labelRecord.synxnet

         labelDir = os.path.join(dstDir, synxnet)

         os.mkdir(labelDir)

     #copy images
     for imageRecord in imageList:

         imageSrcFilePath = imageRecord.imagePath

         imageDstFilePath = "%s/%s/%s"%(dstDir, imageRecord.labelRecord.synxnet, os.path.basename(imageSrcFilePath))

         shutil.copyfile(imageSrcFilePath, imageDstFilePath)


def splitDataSet(srcDir, trainDir, validationDir, sampleType, fraction):

    if(srcDir == trainDir or srcDir == validationDir or trainDir == validationDir):
        print " src dir , train dir , validationDir should not  be the same"
        exit(-1)

    mapLabels = getLabelList(srcDir)

    imageValidateList = []

    imageTrainList    = []

    if(sampleType == "fixed"):
        imageValidateList,imageTrainList = sampleFixedDataset(srcDir, mapLabels, fraction)
    else :
         if(sampleType == "random"):
             imageValidateList,imageTrainList = sampleRandomDatset(srcDir, mapLabels, fraction)
         else:
            print("unsupport sample method  error ", sampleType)
            return

    if(len(imageValidateList) == 0 or len(imageTrainList) == 0):
        print("generate fixed data error ")
        exit(-1)

    copyImageFile(srcDir, mapLabels, trainDir, imageTrainList)

    copyImageFile(srcDir, mapLabels, validationDir, imageValidateList)

    print("labels:", len(mapLabels))
    print("totalimages:", len(imageValidateList) + len(imageTrainList))
    print("validatelist:", len(imageValidateList))
    print("trainlist:", len(imageTrainList))

def useage():
    print "datasetgenerator script can split one dir data set into train and validation dataset."
    print "python dataset.py --srcdir=/xxx --traindir=/cccc  --valdir=/dddd  --fraction=0.2 --type=[fixed|random]"
    print "srcdir: the original dir: "
    print "     images dataset in the srcdir should be orgizned as follows:"
    print "     srcdir/label.txt"
    print "     srcdir/labelid1/1.jpg"
    print "     srcdir/labelid1/2.jpg"
    print "     ..........       "
    print "     srcdir/labelidn/n.jpg"
    print "traindir: the train data set dir"
    print "valdir:   the validation data set dir "
    print "fraction: the percentage of validate dataset in the whole data set"
    print "type: fixed, within every label ,use the same fraction; random , within the whole dataset , use the fraction"


def getEnv():

    if(len(sys.argv) < 4):
        useage()
        exit(-1)

    print __name__ + "start------------------"

    if(G_OPTION.decode(sys.argv) == False):
        print "not enough argvments "
        exit(-1)

    srcDir  = G_OPTION.getValue("srcdir")

    if(srcDir == ""):
        print "please input the src dir which contains the images you wan to convert"
        exit(-1)

    trainDir = G_OPTION.getValue("traindir")

    if(trainDir == ""):
        print "please input the train dir which contains the converted images file"
        exit(-1)

    validationDir = G_OPTION.getValue("valdir")

    if(validationDir == ""):
        print "please input the validation dir which contains the converted images file"
        exit(-1)

    sampleType = G_OPTION.getValue("type")

    if((sampleType == "") or (MAP_SAMPLE_TYPE.get(sampleType) == "")):
        print " the sample type must be in both [fixed|rand] "
        exit(-1)

    fraction = string.atof(G_OPTION.getValue("fraction"))

    if(fraction <=0  or fraction == 1.0):
        print "the percentage of validation and train set is not valid it must be biger than 0 and less than 1"
        exit(-1)

    print "the src images dir : " , srcDir
    print "the train dst dir : ", trainDir
    print "the validation dst dir : ", validationDir

    print __name__ + "end------------------"

    return srcDir, trainDir, validationDir, sampleType, fraction;


def main():
    srcDir, trainDir, validationDir, sampleType, fraction = getEnv()
    splitDataSet(srcDir, trainDir, validationDir, sampleType, fraction)


if __name__ == "__main__":
    main()