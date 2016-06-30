import gzip
import os
import re
import sys
import tarfile
import urllib
import tensorflow as tf
import string


class Option:
    mapOptions = {}

    def __init__(self):
        print "init Option"

    def decode(self, argv):
        if(len(argv) <= 1):
            return False

        argvs = argv[1:];

        for item in argvs:
            itemlist = item.split("--");

            if(len(itemlist) == 2):
                innerList = itemlist[1].split("=");
                if(len(innerList) == 2):
                    print innerList[0], innerList[1]
                    self.mapOptions[innerList[0]] = innerList[1]
                else:
                    print "error data format:", itemlist[1]
            else:
                print " error need data sepstr --",item


    def getValue(self, key):
        return self.mapOptions.get(key)

    def setValue(self, entry):
        self.mapOptions[entry[0]] = self.mapOptions[entry[1]]
