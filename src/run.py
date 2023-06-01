import xmltodict
import sys

import preprocessing
import analysis



# Import annotations from xml file and parse it to dictionary
with open('annotations.xml') as fd:
    doc = xmltodict.parse(fd.read())



if __name__ == '__main__':
    debug_show_plots = True
    output_filename = sys.argv[1]
    enable_visualization = sys.argv[2] == "-v"
    if enable_visualization:
        incisions = sys.argv[3:-1]
    else:
        incisions = sys.argv[2:-1]
    if sys.argv[-1] == "inc":
        inc = sys.argv[-1]
    print(sys.argv)

    pp = preprocessing.Preprocessing(incisions, output_filename, enable_visualization, debug_show_plots) # Generate output.json in preprocessing script
    an = analysis.Analysis(output_filename)

    x = 0