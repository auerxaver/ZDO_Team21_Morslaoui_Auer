import xmltodict
import sys

import preprocessing



# Import annotations from xml file and parse it to dictionary
with open('annotations.xml') as fd:
    doc = xmltodict.parse(fd.read())



if __name__ == '__main__':
    output_filename = sys.argv[1]
    enable_visualization = sys.argv[2]
    incisions = sys.argv[3:-1]
    if sys.argv[-1] == "inc":
        inc = sys.argv[-1]
    print(sys.argv)

    pp = preprocessing.Preprocessing(incisions, output_filename) # Generate output.json in preprocessing script

    x = 0