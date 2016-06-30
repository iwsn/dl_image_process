__author__ = 'willqian'

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
import glob
import random

class LabelRecord:
    def __init__(self, labelId, synxnet, humanName):
        self.labelId   = labelId
        self.synxnet   = synxnet
        self.humanName = humanName

class ImageRecord:
    def __init__(self, imagePath, labelRecord):
        self.imagePath       = imagePath
        self.labelRecord     = labelRecord



def sampleList(imageRecords, fraction):

    imageIndexList = list(xrange(len(imageRecords)))

    sampleLen      = int(len(imageIndexList)*fraction)

    imageValidateIndexList  = random.sample(imageIndexList, sampleLen)

    imageTrainIndexList     = list(set(imageIndexList) - set(imageValidateIndexList))

    imageValidateList       = [imageRecords[i] for i in imageValidateIndexList]

    imageTrainList          = [imageRecords[i] for i in imageTrainIndexList]

    return imageValidateList, imageTrainList

#within every label dir image set to sample
def sampleFixedDataset(srcDir, mapLabels, fraction):

    imageValidateList = []

    imageTrainList    = []

    for labelId in mapLabels:

        labelRecord  = mapLabels[labelId]

        imageFmt     = "%s/%s/*.jpg"%(srcDir, labelRecord.synxnet)

        filePaths    = glob.glob(imageFmt)

        imageRecords = []

        for filePath in filePaths:
            imageRecord = ImageRecord(filePath, labelRecord)
            imageRecords.append(imageRecord)

        imageValidateTmpList, imageTrainTmpList  = sampleList(imageRecords, fraction)

        imageValidateList.extend(imageValidateTmpList)
        imageTrainList.extend(imageTrainTmpList)

    print("labelid:", labelRecord.labelId, labelRecord.synxnet , "image size :", len(imageRecords), " validate size:", len(imageValidateTmpList), " train size: ", len(imageTrainTmpList))

    return imageValidateList, imageTrainList


#widthin the whole image set to sample into train and validate set

def sampleRandomDatset(srcDir, mapLabels, fraction):

    imageRecords = []

    for labelId in mapLabels:

        labelRecord = mapLabels[labelId]

        imageFmt    = "%s/%s/*.jpg"%(srcDir, labelRecord.synxnet)

        filePaths   = glob.glob(imageFmt)

        for filePath in filePaths:
            imageRecord = ImageRecord(filePath, labelRecord)
            imageRecords.append(imageRecord)

    imageValidateList,imageTrainList  = sampleList(imageRecords, fraction)

    print("image size :", len(imageRecords), " validate size:", len(imageValidateList), " train size: ", len(imageTrainList))

    return imageValidateList, imageTrainList

