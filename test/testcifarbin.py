from PIL import Image

import os
import re
import sys
import struct
import numpy as np

IMAGE_WIDTH = 32
IMAGE_HEIGHT=32
IMAGE_CHANNEL=3

def decodeCifar(fileName):

    file=open(fileName, 'rb').read();

    print fileName

    image_buffer_size = 32*32*3

    index=0

    offset=0;

    while(index < 500):

        label = struct.unpack_from("B", file, offset)

        offset = offset + 1

        print label[0]

        image = []

        fmt = str(image_buffer_size)+"B"

        print struct.unpack_from(fmt, file, offset)

        image.append(struct.unpack_from(fmt, file, offset))

        print np.array(image)

        offset = offset + image_buffer_size

        imagedata = np.transpose(np.uint8(np.array(image)).reshape([3, 32, 32]), [1, 2, 0])

        print imagedata

        # imagedata = np.uint8(np.array(image)).reshape([3, IMAGE_SIZE, IMAGE_SIZE]);
        data = Image.fromarray(imagedata)

        data.save("/home/mqq/willqian/FlowerRecognize/test/"+str(index)+".jpg")

        index=index+1


def testCifarBin():
    if(len(sys.argv) < 2):
        print "please specify the file you want to decode"
        exit(-1)

    filePath = sys.argv[1];

    decodeCifar(filePath)


if __name__ == "__main__":
    testCifarBin()


