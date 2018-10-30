# ###########################################
# Analyse.py
# An analysis framework for feasibility of application of Machine Learning Algorithms
# Currently limited to test feasibility of image classification problems.
# @author Rijul Gupta
# @since 28 October 2018

import argparse
from subprocess import Popen, PIPE, STDOUT
import os
import shutil
import re
import matplotlib.pyplot as plt

DATAPATH = ''
"""
str: DATAPATH
The basepath where to get the data from
"""

BASEPATH = ''
"""
str: BASEPATH
The basepath where to work on the data used for analysis
"""

MODELSPATH = ''
"""
str: MODELSPATH
The path where to store the models information
"""

SEGMENTSPATH = ''
"""
str: SEGMENTSPATH
The path where to store the segments
"""

RESULTSPATH = ''
"""
str: RESULTSPATH
The path where to store the results
"""

SEGMENTSIZE = 100
"""
int: SEGMENTSIZE
The minimum size of one segment
"""

SEGMENTS = []
"""
array: SEGMENTS
Segments of data to be tested
"""

SEGMENTSIZES = []
"""
array: SEGMENTSIZES
Sizes of the segments of data to be tested
"""

MODULESPATH = './modules.csv'
"""
str: MODULESPATH
The path of the file that contains the modules that should be used for analysis
"""

MODULES = []
"""
array: MODULES
Modules to test feasibility with
"""

RESULTS = []
"""
array: RESULTS
The results of the analysis
"""

TRAININGSTEPS = 0
"""
int: TRAININGSTEPS
The number of steps to train for
"""

def createTemp():
    """
    creates a temp directory for working on the analysis
    """
    global BASEPATH, SEGMENTSPATH, MODELSPATH, RESULTSPATH
    BASEPATH = os.path.join(DATAPATH,'.temp')
    SEGMENTSPATH = os.path.join(BASEPATH,'segments')
    MODELSPATH = os.path.join(BASEPATH,'models')
    RESULTSPATH = os.path.join(BASEPATH,'results')
    '''Cleaning old files'''
    if os.path.exists(BASEPATH) and os.path.isdir(BASEPATH):
        shutil.rmtree(BASEPATH)
    '''Preparing folders for new analysis'''
    os.mkdir(BASEPATH)
    os.mkdir(SEGMENTSPATH)
    os.mkdir(MODELSPATH)
    os.mkdir(RESULTSPATH)

def getDirs(dir):
    """
    retrieves directories from given directory
    """
    return [name for name in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, name))]

def getDataDirs():
    """
    retrieves the data directories from DATAPATH
    """
    dirs = getDirs(DATAPATH)
    dirs.remove('.temp')
    return dirs

def getFiles(dir):
    """
    retrieves files from a given directory
    """
    dir = os.path.join(DATAPATH, dir)
    return [name for name in os.listdir(dir)
            if os.path.isfile(os.path.join(dir, name))]

def getNumFiles(dir):
    """
    retrieves the number of files in a given directory
    """
    return len(getFiles(dir))

def segmentData():
    """
    rearranges the data into segments for subsequent testing
    """
    global SEGMENTS
    datadirs = getDataDirs()
    for dir in datadirs:
        numFiles = getNumFiles(dir)
        if ('minSegment' not in locals() or numFiles < minSegment):
            minSegment = numFiles
    segments = (minSegment // SEGMENTSIZE) + 1
    for segment in range(segments-1, -1, -1):
        segmentPath = os.path.join(SEGMENTSPATH,'segment'+str(segment))
        os.mkdir(segmentPath)
        SEGMENTSIZES.append(minSegment // (segment+1))
        for dir in datadirs:
            files = getFiles(dir)
            dirPath = os.path.join(segmentPath,dir)
            os.mkdir(dirPath)
            for num in range(minSegment // (segment+1)):
                file = files[num]
                src = os.path.join(DATAPATH, dir, file)
                dst = os.path.join(dirPath, file)
                shutil.copy(src, dst)
        SEGMENTS.append(segmentPath)

def loadModules():
    """
    loads the modules to test for feasibility
    """
    global MODULES
    with open(MODULESPATH) as file:
        lines = [line.rstrip('\n') for line in file]
    MODULES = lines

def analyseData(segment = None, module = None):
    """
    analyses the given data for given module

    Args:
        segment (str): Path of the segment to test with
        module (str): Identifier of the module to test with
    
    Returns:
        object: result of the analysis
    """
    modName = re.sub('[^A-Za-z0-9]+', '', module)
    key = modName + '_' + os.path.basename(segment)
    bottleneckDir = os.path.join(BASEPATH, '.bottleneck', key)
    graphDir = os.path.join(MODELSPATH, key)
    segModDir = os.path.join(RESULTSPATH, key)
    summariesDir = os.path.join(segModDir, 'summary')
    modelsDir = os.path.join(segModDir, 'model')
    logDir = os.path.join(BASEPATH, '.logs')
    logfile = os.path.join(logDir, key+'.out.log')
    if not os.path.exists(logDir):
        os.makedirs(logDir)
    log = open(logfile, 'a+')

    cmd = [ 'python', 'retrain.py',
            '--image_dir', segment,
            '--tfhub_module', module,
            '--summaries_dir', summariesDir,
            '--bottleneck_dir', bottleneckDir,
            '--saved_model_dir', modelsDir,
            '--output_graph', graphDir
        ]
    if (TRAININGSTEPS != 0):
        cmd.append('--how_many_training_steps')
        cmd.append(str(TRAININGSTEPS))
    return Popen(cmd, stdout=log, stderr=STDOUT)

def processResults():
    """
    Parses and stores the results
    """
    logDir = os.path.join(BASEPATH, '.logs')
    logs = getFiles(logDir)
    results = [['Model', 'Segment Size', 'Accuracy', 'N']]
    accuracies = []
    for index, log in enumerate(logs):
        modelName = log.split('_')[0]
        file = open(os.path.join(logDir, log), 'r')
        for line in file:
            if re.search('^.*Final test accuracy.*', line):
                match = re.match(r'.* = (.*)\% \(N=(.*)\)', line)
                accuracy = match.group(1)
                N = match.group(2)
                result = [modelName, SEGMENTSIZES[index], accuracy, N]
                accuracies.append(accuracy)
                results.append(result)
    csvString = '\n'.join([','.join(str(v) for v in x) for x in results])
    resultsDir = os.path.join(BASEPATH, 'results')
    resultsFile = os.path.join(resultsDir, 'result.csv')
    file = open(resultsFile, 'w+')
    file.write(csvString)
    plotFile = os.path.join(resultsDir, 'results.png')
    plt.xlabel('$Segments$ $(number of items)$')
    plt.ylabel('$Accuracy$ $(%)$')
    plt.grid(False)
    plt.plot(SEGMENTSIZES, accuracies, 'r-')
    plt.savefig(plotFile)

def prepareData():
    """
    prepares the data for testing
    """
    createTemp()
    segmentData()

def main():
    """
    main function to start the analysis
    """
    prepareData()
    loadModules()
    processes = []
    for segment in SEGMENTS:
        for module in MODULES:
            processes.append(analyseData(segment, module))
    print('total processes running: '+str(len(processes)))
    for index, process in enumerate(processes):
        print('waiting for process '+str(index))
        process.wait()
    processResults()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--modules',
        type=str,
        default='',
        help='Path to file that contains modules that should be used.'
    )
    parser.add_argument(
        '--trainingSteps',
        type=int,
        default='',
        help='No of steps to be used for training.'
    )
    parser.add_argument(
        '--segmentSize',
        type=int,
        default='',
        help='Minimum size of segments to be used.'
    )
    flags, unparsed = parser.parse_known_args()
    if not flags.image_dir:
        raise Exception('Must set flag --image_dir.')
    if flags.modules:
        MODULESPATH = flags.modules
    if flags.trainingSteps:
        TRAININGSTEPS = flags.trainingSteps
    if flags.segmentSize:
        SEGMENTSIZE = flags.segmentSize
    DATAPATH = os.path.abspath(flags.image_dir)
    main()