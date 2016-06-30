from PIL import Image


from config import *
from convertmodelinput.convert_to_resnet import *


def showImage(fileName):

    file=open(fileName, 'rb').read();

    image_buffer_size = IMAGE_HEIGHT*IMAGE_WIDTH*IMAGE_CHANNEL

    index=0

    offset=0;

    while(index < 10000):

        label = struct.unpack_from(">1B", file, offset)

        offset = offset + 1;

        image = []

        pix_index = 0

        while(pix_index < image_buffer_size):
            fmt = ">1B";
            image.append((struct.unpack_from(fmt, file, offset)[0]))
            offset = offset + 1
            pix_index=pix_index+1

        imagedata = np.transpose(np.uint8(np.array(image)).reshape([3, IMAGE_HEIGHT, IMAGE_HEIGHT]), [1, 2, 0]);

        # imagedata = np.uint8(np.array(image)).reshape([3, IMAGE_SIZE, IMAGE_SIZE]);
        data = Image.fromarray(imagedata);

        data.show();

        index=index+1

def test(dstDir):
     fileDir = os.listdir(dstDir)

     for fileItem in fileDir:

        filePath = os.path.join(dstDir , str(fileItem))

        showImage(filePath)