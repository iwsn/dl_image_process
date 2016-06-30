from convertmodelinput.converttocifar import *
from convertmodelinput.converttoinception import *
from config import *
from util.Option import *

#parse env path

G_OPTION = Option()

#read file parse lable index and label name
#read label dir and label index
def parseLabelPath(labelPath):

    #labeldir,
    nameDict= {}
    idDict  = {}


    file = open(labelPath, "rb")

    index = 0;

    for line in file:

        line = line.strip("\n")

        if(line == ""):
            continue

        items = line.split("|")

        if(len(items) < 2):
            print " label text file error:", line
            exit(-1)

        #line no as label index
        #labeldir and label text

        #line format labelid | labelname

        labelId  = index

        labelDir = items[0]

        labelName= items[1]

        idDict[labelId]   = labelDir

        nameDict[labelDir] = labelName

        index=index+1

    return idDict, nameDict


#find all label image
def findImageFiles(idDict, srcDir):

    imagePaths = []

    imageLabels=[]

    for labelId in idDict:

        labelDir = idDict[labelId];

        mactchMode = "%s/%s/*.jpg"%(srcDir, labelDir)

        tmpMatchFiles = glob.glob(mactchMode)

        imagePaths.extend(tmpMatchFiles)

        imageLabels.extend([labelId]*len(tmpMatchFiles))

    shuffleIndex = range(len(imagePaths));
    random.seed(time.time())
    random.shuffle(shuffleIndex)

    imagePaths = [imagePaths[i] for i in shuffleIndex]
    imageLabels= [imageLabels[i] for i in shuffleIndex]

    return imagePaths, imageLabels

#convert images to tensorflow demo cnn model input files
def convert(type, dataName, srcDir, dstDir, batch):

    labelPath = os.path.join(srcDir, "label.txt")

    if(os.path.exists(labelPath) == False):
        print " src dir does contain label.txt file error :",srcDir
        exit(-1)

    idDict, nameDict = parseLabelPath(labelPath)

    print idDict
    print nameDict

    imagePaths, imageLabels = findImageFiles(idDict, srcDir)

    if(type == "cifar"):
        convertToCifar(imagePaths, imageLabels, dstDir, batch)
    else:
        if(type == "inception"):
            convertToInception(dataName, idDict, nameDict, imagePaths, imageLabels, dstDir, batch)
        else:
            print(" unknow support type error:"+type)



def useage():
    print "convert program can be used to convert our own images format to the tensorflow cnn network format for traing,"
    print "so we can quickly debug tensorflow demo cnn network on our own data."
    print "python convert.py --type=[cifar|inception] --dataname=[train|validation] --srcdir= --dstdir=  --batch=6 --size=10000 "
    print  "type: represent to convert current format to type cnn format "
    print  "dataname: reopreset data is train or validation data set"
    print  "srcdir: reprent to contain src images and label txt. it should  orginaize as follows:"
    print  " srcdir/label.txt"
    print  " srcdir/labelid1/1.jpg"
    print  " srcdir/labelid1/2.jpg"
    print  "     ..........       "
    print  " srcdir/labelidn/n.jpg"
    print  "dstdir : represent to the type format wo want to convert"
    print  "batch  : represent the desired files num"



def getEnv():

    if(len(sys.argv) < 6):
        useage()
        exit(-1)

    print __name__ + "start------------------"

    if(G_OPTION.decode(sys.argv) == False):
        print "not enough argvments "
        exit(-1)

    type   = G_OPTION.getValue("type");

    if(type == ""):
        print " please input the demo cnn type you want to convert to "
        exit(-1)


    if(MAP_DEMO_CNN_TYPE.get(type, "") == ""):
        print " the support  list is ", MAP_DEMO_CNN_TYPE.keys() ,"the type : ", type, " is not support so far"
        exit(-1)

    dataName = G_OPTION.getValue("dataname")


    srcDir  = G_OPTION.getValue("srcdir")

    if(srcDir == ""):
        print "please input the src dir which contains the images you wan to convert"
        exit(-1)

    dstDir = G_OPTION.getValue("dstdir")

    if(dstDir == ""):
        print "please input the dst dir which contains the converted images file"
        exit(-1)

    batch = string.atoi(G_OPTION.getValue("batch"))

    print "the desired type you want to convert : ", type
    print "the src images dir : " , srcDir
    print "the dst dir : ", dstDir


    print __name__ + "end------------------"

    return type, dataName, srcDir, dstDir, batch;


def main():
    type, dataName, srcDir , dstDir, batch = getEnv()
    convert(type, dataName, srcDir, dstDir, batch)
    #test(dstDir)

if __name__ == "__main__":
    main()