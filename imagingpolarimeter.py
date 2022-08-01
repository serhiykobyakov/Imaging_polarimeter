"""
Data processing module for Imaging Polarimeter

University of Cardinal Stefan Wyszyński in Warsaw (Poland)
Institute of Physical Sciences

"""

from cmath import pi
from genericpath import isdir
import sys
import glob
import os
import configparser
import math
from PIL import Image
import numpy as np
from numpy import sqrt
#import scipy as sp
#from scipy import ndimage
import subprocess
#from matplotlib import pyplot as plt


__author__ = "Serhiy Kobyakov"
__authors__ = ["Serhiy Kobyakov"]
__copyright__ = "Copyright 2022, Serhiy Kobyakov"
__credits__ = ["Yaroslav Shopa"]
__date__ = "2022/01/08"
__deprecated__ = False
__license__ = "MIT"
__maintainer__ = "Serhiy Kobyakov"
__status__ = "Production"
__version__ = "2022.08.01"


class imagingpolarimeter:
    # create 16bit images project?
    # for testing purpouses, must be False after release
    write16bit = True

    # the file with the experiment variables
    variablesFName = 'variables.m'

    # the file with the calibrated positions of the Analizator
    xarrayFName = 'APosCalibrated.dat'

    # experiment statistics file name
    statFName = '0_info.txt'

    # RAW files extention:
    raw_extension = 'CR2'

    # sigma value of gaussian filter for smoothing images
    # It is 1 for Canon EOS M3
    theSigma = 1

    # threshold to decide if the fit is actually a good parabola
    signal_compare = 0.6    


    def __init__(self, dataDir) -> None:
        self.dataDir = dataDir
        # check if there is variables.m file
        if not os.path.isfile(os.path.join(self.dataDir, self.variablesFName)):
            print("No " + self.variablesFName + " file found in " + self.dataDir + "\nAborting!")
            sys.exit()
        # check if there is variables.m file
        if not os.path.isfile(os.path.join(self.dataDir, self.xarrayFName)):
            print("No " + self.xarrayFName + " file found in " + self.dataDir + "\nAborting!")
            sys.exit()
        print("")
        self._getallvariables()
        if self.write16bit:
            if not os.path.isdir("16bit"):
                cmd = "mkdir 16bit >/dev/null 2>&1"
                os.system(cmd)
        self._unpackRAW('raw_background_images')
        self._unpackRAW('raw_images')
        self._mkdark()
        self._maketheimages()

        if self.write16bit:
            cmd = "cp " + self.variablesFName + " 16bit >/dev/null 2>&1"
            os.system(cmd)
            cmd = "cp " + self.xarrayFName + " 16bit >/dev/null 2>&1"
            os.system(cmd)
            cmd = "cp 0_info.txt 16bit >/dev/null 2>&1"
            os.system(cmd)
            cmd = "cp log.txt 16bit >/dev/null 2>&1"
            os.system(cmd)


    def _writeVarToFile(self, theStr):
    # write the line theStr to the file "variables.m"
    # use it to save experiment variables
        try:
            with open(self.variablesFName, "a") as f:
                f.write(theStr + '\n')
                f.close()
        except IOError:
            print('problem opening variables.m!')


    def _getallvariables(self):
        # let's read the variables:
        print("Reading variables..", end='', flush=True)
        config = configparser.ConfigParser()
        config.read(self.variablesFName)

        self.AllImages = config.getint('Experiment', 'AllImages')
        print(".", end='', flush=True)

        if self.AllImages != config.getint('Experiment', 'NImagCaptured'):
            print("\nImages captured is not equal to estimated")
            print("Incomplete experiment data or error in the " + self.variablesFName + " file?")
            print("Aborting!")
            sys.exit()
        self.theDate = config.get('Experiment', 'theDate')
        self.theSample = config.get('Experiment', 'theSample')
        self.theMeasurementType = config.get('Experiment', 'theMeasurementType')
        self.NdarkImages = config.getint('Experiment', 'NdarkImages')
        self.Nimages = config.getint('Experiment', 'Nimages')
        self.NPSteps = config.getint('Experiment', 'NPSteps')
        self.NASteps = config.getint('Experiment', 'NASteps')
        self.AStepRad = config.getfloat('Experiment', 'AStepRad')
        print(".", end='', flush=True)

        if os.path.isfile(self.xarrayFName):
            self.xarray = np.loadtxt(self.xarrayFName)
        else:
            print('no APosCalibrated.dat file found!')
        if self.NASteps != self.xarray.size:
            print("\nError!")
            print("Analizator positions (" + str(self.NASteps) + ") in file " + self.variablesFName)
            print("is not equal to calibrated positions (" + str(self.xarray.size) + ") found in " + self.xarrayFName)
            print("Aborting!")
            sys.exit()

        # write measurement info to file
        try:
            with open(self.statFName, "a") as f:
                f.write('Experiment     : ' + self.theSample + '\n')
                f.write('Experiment type: ' + self.theMeasurementType + '\n')
                f.write(self.theDate + '\n\n')
                f.write('Polarizator steps: ' + str(self.NPSteps) + '\n')
                f.write('Analizator steps: ' + str(self.NASteps) + '\n')
                f.write('Images averaged: ' + str(self.Nimages) + '\n')
                f.write('Dark images averaged: ' + str(self.NdarkImages) + '\n\n')
                f.close()
        except IOError:
            print('EE problem opening ' + self.statFName +'!')       

        self.imgw, self.imgh = 0, 0
        self.theColor = config.get('Experiment', 'theColor', fallback="")
        if len(self.theColor) == 0: self._checkImgParameters()
        if self.imgw == 0: self.imgw = config.getint('Experiment', 'imgw')
        if self.imgh == 0: self.imgh = config.getint('Experiment', 'imgh')
        print(".", end='', flush=True) 

        print("done", flush=True)

    
    def _checkImgParameters(self) -> str:
        filesList=glob.glob(os.path.join(self.dataDir, 'raw_images', '') + "*." + self.raw_extension)
        if len(filesList) > 0:   # if there are raw files
            filesList.sort(key=os.path.getmtime)
            cmd="/usr/bin/4channels " + str(filesList[0]) + " >/dev/null 2>&1"
            os.system(cmd)
            fname = filesList[0]+"."+"R"+".tiff"
            print(".", end='', flush=True)

            # Checking image parameters
            self.imgw, self.imgh = 0, 0
            self.imgw, self.imgh = Image.open(fname).size
            if self.imgw > 0: self._writeVarToFile("imgw=" + str(self.imgw))
            else:
                print("\nError!")
                print("Can't read image width!")
                print("Aborting!")
                sys.exit()
            if self.imgh > 0: self._writeVarToFile("imgh=" + str(self.imgh))
            else:
                print("\nError!")
                print("Can't read image height!")
                print("Aborting!")
                sys.exit()
            print(".", end='', flush=True)

            # Checking the channel we use
            cmd="identify -format \"%[mean]\" " + filesList[0]+"."+"R"+".tiff"
            mR = float(subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()[0])
            cmd="identify -format \"%[mean]\" " + filesList[0]+"."+"G"+".tiff"
            mG = float(subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()[0])
            cmd="identify -format \"%[mean]\" " + filesList[0]+"."+"B"+".tiff"
            mB = float(subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()[0])
            themean={'R':mR,'G':mG,'B':mB}
            self.theColor=max(themean, key=themean.get)
            self._writeVarToFile("theColor=" + self.theColor)
            print(".", end='', flush=True)

            # write camera info into file
            try:
                with open(self.statFName, "a") as f:
                    cmd="exiftool -s -s -s -Model " + filesList[-1]
                    f.write('Camera model: ' + subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()[0].decode("utf-8"))
                    f.write('Single-channel image size: ' + str(self.imgw) + 'x' + str(self.imgh) + '\n\n')
                    f.close()
            except IOError:
                print('EE problem opening ' + self.statFName +'!')  

            # remove unnecessary tiff files
            for f in glob.glob(os.path.join(self.dataDir, 'raw_images', '') + "*.tiff"):
                os.remove(f)
        else:
            print("\nError!")
            print("No dir: " + os.path.join(self.dataDir, 'raw_images', ''))
            print("Aborting!")
            sys.exit()


    def _unpackRAW(self, intheDir):
        filesList=glob.glob(os.path.join(self.dataDir, intheDir, '') + "*." + self.raw_extension)
        if len(filesList) > 0:   # if there are raw files
            filesList.sort(key=os.path.getmtime)
            print("Converting RAW to tif in /" + intheDir + ".", end='', flush=True)
            for j in range(len(filesList)):
                cmd="/usr/bin/4channels -B " + str(filesList[j]) + " >/dev/null 2>&1"
                os.system(cmd)
                if os.path.isfile(filesList[j] + "." + self.theColor + ".tiff"):
                    os.rename(filesList[j] + "." + self.theColor + ".tiff", filesList[j] + "." + self.theColor + ".tif")
                    os.remove(filesList[j]) # remove the raw file
                for f in glob.glob(os.path.join(intheDir, '') + "*.tiff"):
                    os.remove(f)  # remove all .tiff files
                print(".", end='', flush=True)          
            print("done")  # unpacking raw files


    def _mkdark(self):
    # assemble the dark image
        if not os.path.isfile('dark.tiff') and os.path.isdir(os.path.join(os.getcwd(), 'raw_background_images')):
            print('Assembling the dark image:')
            # get list of tif files in the bg directory
            bgfilesList=glob.glob(os.path.join(os.getcwd(), 'raw_background_images', '*.tif'))
            bgfilesList.sort(key=os.path.getmtime)            
            
            # read the dark images to array
            print("    Reading " + str(len(bgfilesList)) +" dark images into array..", end='', flush=True)
            darkarr=np.array([np.array(Image.open(fname)) for fname in bgfilesList])
            print(".done")
            n_z, n_y, n_x = darkarr.shape   # get dimensions of the array
            print("    The array shape: x=", n_x, " y=", n_y, "z=", n_z, " size in memory: "+ str(round(darkarr.nbytes / 1024 / 1024,2)) + " Mb")

            # open file for statistical information about background images
            arr = np.zeros((self.imgh, self.imgw), dtype=np.float32)
            arr = np.array(Image.open(bgfilesList[0]), dtype=np.float32)

            print("    Sigma clipping the dark images.", end='', flush=True)
            for z in range(n_z):
                subarr=darkarr[z, 0:n_y, 0:n_x]  # slice the image from array
                theSigma=np.std(subarr)
                theoldSigma=3*theSigma
                theMedian=np.median(subarr)
                while (theoldSigma - theSigma)/theSigma > 1e-10:
                    theoldSigma=theSigma
                    subarr[abs(subarr - theMedian) > 3*theSigma] = theMedian    
                    theSigma=np.std(subarr)
                    theMedian=np.median(subarr)
                darkarr[z, 0:n_y, 0:n_x] = subarr  # put the filtered data back to the array
                print(".", end='', flush=True)
            print("done")

            # write image info to file
            # using the last image which is still in subarr
            try:
                with open(self.statFName, "a") as f:
                    f.write('Single-channel out-of-camera image statistics:\n')
                    f.write("  Average: {:8.3f}\n".format(np.average(subarr)))
                    f.write("  Median:  {:8.3f}\n".format(np.median(subarr)))
                    f.write("  stdev:   {:8.5f}\n".format(np.std(subarr)))
                    f.write("  Noise:   {:8.5f}\n\n".format(3*np.std(subarr)))
                    f.close()
            except IOError:
                print('EE problem opening ' + self.statFName +'!')

            print("    Averaging " + str(len(bgfilesList)) +" dark images into singe image..", end='', flush=True)
            darkimg=np.zeros((n_y, n_x), dtype=np.float32)
            for z in range(n_z):
                darkimg = darkimg + darkarr[z, 0:n_y, 0:n_x]
            darkimg=darkimg/n_z

            # Noise level of the averaged images
            self.imageNoise = 3*np.std(darkimg)

            # write dark image info to file
            # using the last image which is still in darkimg
            try:
                with open(self.statFName, "a") as f:
                    f.write('The dark image statistics:\n')
                    f.write("  Average: {:8.3f}\n".format(np.average(darkimg)))
                    f.write("  Median:  {:8.3f}\n".format(np.median(darkimg)))
                    f.write("  stdev:   {:8.5f}\n".format(np.std(darkimg)))
                    f.write("  Noise:   {:8.5f}\n\n".format(self.imageNoise))
                    f.close()
            except IOError:
                print('EE problem opening ' + self.statFName +'!')

            # save the dark file
            np.save("dark.npy", darkimg)
            
            if self.write16bit:
            # save the dark image in tiff format
                im = Image.fromarray(np.uint16(np.around(darkimg)), mode='I;16')
                im.save("16bit/dark.tiff", "TIFF")

            # remove unnecessary tiff files
            if os.path.isfile('dark.npy'):
                for f in glob.glob(os.path.join(os.getcwd(), 'raw_background_images','*.tif')):
                    os.remove(f)
                os.removedirs(os.path.join(os.getcwd(), 'raw_background_images'))

            print(".done")  # working on the dark image


    def _maketheimages(self):
    # make the images for the experiment
        filesList=glob.glob(os.path.join(self.dataDir, 'Images', '???_???.tiff'))
        if len(filesList) != self.NASteps*self.NPSteps:
            print('\nAssembling images:')
            # open the dark image
            if os.path.isfile("dark.npy"):
                dark = np.load("dark.npy")
            else:
                print("EE No dark image found!")
                sys.exit()

            # make the directory for images if it does not exists
            theDir = os.path.join(self.dataDir, 'Images')
            the16bitDir = os.path.join(self.dataDir, '16bit', 'Images')
            if not os.path.isdir(theDir):
                cmd="mkdir " + theDir + " >/dev/null 2>&1"
                os.system(cmd)
            if self.write16bit:
                if not os.path.isdir(the16bitDir):
                    cmd = "mkdir " + the16bitDir + " >/dev/null 2>&1"
                    os.system(cmd)

            filesList=glob.glob(os.path.join(self.dataDir, 'raw_images', '*.tif'))
            filesList.sort(key=os.path.getmtime)

            # make images for each Analizator position
            print("    Making images for each Analizator position.", end='', flush=True)
            for i in range(self.NPSteps):
                for j in range(self.NASteps):
                    filesInAngle=filesList[(i*self.NASteps +j)*self.Nimages:(i*self.NASteps +j)*self.Nimages+self.Nimages]
                    newFName = str(format(i+1, '03d')) + "_" + str(format(j+1, '03d'))                    
                    """
                    print('\n\n** i=' + str(i) + ' j=' + str(j))
                    print('** files in angle:', len(filesInAngle))
                    print('** files:', filesInAngle)
                    print('** files from:', (i*self.NASteps +j)*self.Nimages)
                    print('** files to  :', (i*self.NASteps +j)*self.Nimages+self.Nimages)                
                    print('** out file:', newFName)
                    """
                    arr = np.zeros((self.imgh, self.imgw), dtype=np.float32)
                    for n in range(len(filesInAngle)):
                        fname=filesInAngle[n]  #+"."+theColor+".tiff"
                        arr = arr + np.array(Image.open(fname), dtype=np.float32)
                        print('.', end='', flush=True)
                    arr = arr/len(filesInAngle) - dark                   
                    np.save(theDir + '/' + newFName, arr)
                    print('.', end='', flush=True)
                    if self.write16bit:
                        im = Image.fromarray(np.uint16(np.around(np.where(arr < 0., 0., arr))), mode='I;16')
                        im.save(os.path.join(the16bitDir, newFName + ".tiff"), "TIFF")  # save the dark image  
            print("done")

            # remove unnecessary tiff files
            if len(glob.glob(os.path.join(theDir, '*.npy'))) == self.NPSteps*self.NASteps:
                for f in glob.glob(os.path.join(os.getcwd(), 'raw_images', '*.tif')):
                    os.remove(f)
                os.removedirs(os.path.join(os.getcwd(), 'raw_images'))

            print('')



    def PrintStatus(self):
    # just print variable values
    # for testing purpouses, must be removed in the beta version
        print("\nWorking dir:", self.dataDir)
        print("")
        print("  theSample: \"" + self.theSample + "\"")
        print("  theMeasurementType: \"" + self.theMeasurementType + "\"")
        print("  NdarkImages:", self.NdarkImages)
        print("  NImages:", self.Nimages)
        print("  AllImages:", self.AllImages)
        print("")
        print("")
        print("  NPSteps:", self.NPSteps)
        print("  NASteps:", self.NASteps)
        print("  AStepRad:", self.AStepRad)
        print("")
        print("")
        print("  color: \"" + self.theColor + "\"")
        print("")

