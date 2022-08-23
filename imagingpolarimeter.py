"""
Data processing module for Imaging Polarimeter

University of Cardinal Stefan Wyszyński in Warsaw (Poland)
Institute of Physical Sciences

"""

from cmath import nan, pi
from genericpath import isdir, isfile
import sys
import glob
import os
import configparser
import pickle
import math
from PIL import Image
import numpy as np
from numpy import sqrt
import scipy as sp
from scipy import ndimage
import subprocess
from matplotlib import pyplot as plt
import multiprocessing

from scipy.optimize import leastsq
#from scipy.optimize import least_squares

import time


__author__ = "Serhiy Kobyakov"
__authors__ = ["Serhiy Kobyakov"]
__copyright__ = "Copyright 2022, Serhiy Kobyakov"
__credits__ = ["Yaroslav Shopa"]
__date__ = "2022.08.23"
__deprecated__ = False
__license__ = "MIT"
__maintainer__ = "Serhiy Kobyakov"
__status__ = "Production"
__version__ = "2022.08.23"



class imagingpolarimeter:
    # for testing purpouses:
    decrRadius = 40

    # create 16bit images project?
    # for testing purpouses, must be False after release
    write16bit = False

    # the file with the experiment variables
    varFName = 'projVar.pkl'

    # the file with the experiment variables from data acquisition software
    variablesFName = 'variables.m'

    # the file with the calibrated positions of the Analizator
    xarrayFName = 'APosCalibrated.dat'

    # experiment statistics file name
    statFName = '0_info.txt'

    # RAW files extention:
    raw_extension = 'CR2'

    # max saturation level for image sensor
    #maxI = 800.0

    # sigma value of gaussian filter for smoothing images
    # It is 1 for Canon EOS M3
    # theSigma = 3

    # threshold to decide if the fit is actually a good parabola
    signal_compare = 0.6   

    # create empty dictionary for project variables
    var = {}


    def __init__(self, somePath) -> None:
        # get the directiry the program have been started        
        self.dataDir = os.getcwd()

        # get number of threads on this machine
        self.NThreads = multiprocessing.cpu_count()

        if len(somePath) == 0 and os.path.isfile(self.varFName):
            print(self.varFName + " has not been mentioned but we found it!")
            somePath = os.path.join(os.getcwd(), self.varFName)

        if os.path.isfile(somePath):
            if os.path.splitext(somePath)[1] == '.pkl':
                # if there is file with variables - just load it
                with open(somePath, 'rb') as f:
                    self.var = pickle.load(f)
            else:
                print("\nEE " + somePath + " is not *.pkl file!\nAbort!\n")
                sys.exit()
        else:
        # if no file found - get all necessary variables from raw data
            #self.dataDir = somePath
            # check if there is variables.m file
            if not os.path.isfile(os.path.join(self.dataDir, self.variablesFName)):
                print("No " + self.variablesFName + " file found in " + self.dataDir + "\nAbort!")
                sys.exit()
            # check if there is variables.m file
            if not os.path.isfile(os.path.join(self.dataDir, self.xarrayFName)):
                print("No " + self.xarrayFName + " file found in " + self.dataDir + "\nAbort!")
                sys.exit()
            self._getVariables()

            filesList=glob.glob(os.path.join(self.dataDir, 'raw_background_images', '') + "*." + self.raw_extension)
            if len(filesList) > 0:   # if there are raw files
                filesList.sort(key=os.path.getmtime)
                print("Converting RAW to tif in /" + 'raw_background_images' + ".", end='', flush=True)
                pool = multiprocessing.Pool(self.NThreads)
                pool.map(self._unpRAW, filesList)
                pool.close()
                pool.join()
                print("done")  # unpacking raw files

            filesList=glob.glob(os.path.join(self.dataDir, 'raw_images', '') + "*." + self.raw_extension)
            if len(filesList) > 0:   # if there are raw files
                filesList.sort(key=os.path.getmtime)
                print("Converting RAW to tif in /" + 'raw_images' + ".", end='', flush=True)
                pool = multiprocessing.Pool(self.NThreads)
                pool.map(self._unpRAW, filesList)
                pool.close()
                pool.join()
                print("done")  # unpacking raw files

            self._mkdark()
            self._maketheimages()
            self._findTheCenter()
            self._saveVariables()

            if self.write16bit:
                cmd = "cp " + self.variablesFName + " 16bit >/dev/null 2>&1"
                os.system(cmd)
                cmd = "cp " + self.xarrayFName + " 16bit >/dev/null 2>&1"
                os.system(cmd)
                cmd = "cp 0_info.txt 16bit >/dev/null 2>&1"
                os.system(cmd)
                cmd = "cp log.txt 16bit >/dev/null 2>&1"
                os.system(cmd)
            print("")


    def makedata(self):
        if self.var['theMeasurementType'] == 'characterization':
            self._blurAllImages()
            self._mkCharacterization()


    def _saveVariables(self):
        if not os.path.isfile(self.varFName):
            with open(self.varFName, 'wb') as f:
                pickle.dump(self.var, f, protocol=-1)


    def _getVariables(self):
        # get all the necessary variables
        print("\nReading Imaging Polarimeter variables..", end='', flush=True)        
        config = configparser.ConfigParser()
        config.read(self.variablesFName)

        self.var['AllImages'] = config.getint('Experiment', 'AllImages')
        print(".", end='', flush=True)

        if self.var['AllImages'] != config.getint('Experiment', 'NImagCaptured'):
            print("\nImages captured is not equal to estimated")
            print("Incomplete experiment data or error in the " + self.variablesFName + " file?")
            print("Abort!")
            sys.exit()

        self.var['theDate'] = config.get('Experiment', 'theDate')
        self.var['theSample'] = config.get('Experiment', 'theSample')
        self.var['theMeasurementType'] = config.get('Experiment', 'theMeasurementType')
        self.var['Laser'] = config.get('Experiment', 'Laser')
        self.var['NdarkImages'] = config.getint('Experiment', 'NdarkImages')
        self.var['Nimages'] = config.getint('Experiment', 'Nimages')
        self.var['NPSteps'] = config.getint('Experiment', 'NPSteps')
        self.var['NASteps'] = config.getint('Experiment', 'NASteps')
        self.var['AStepRad'] = config.getfloat('Experiment', 'AStepRad')
        print(".", end='', flush=True)

        if os.path.isfile(self.xarrayFName):
            self.var['xarray'] = np.loadtxt(self.xarrayFName)
        else:
            print('no APosCalibrated.dat file found!')
        if self.var['NASteps'] != self.var['xarray'].size:
            print("\nError!")
            print("Analizator positions (" + str(self.var['NASteps']) + ") in file " + self.variablesFName)
            print("is not equal to calibrated positions (" + str(self.var['xarray'].size) + ") found in " + self.xarrayFName)
            print("Abort!")
            sys.exit()

        # write measurement info to file
        if not os.path.isfile(self.statFName):
            try:
                with open(self.statFName, "w") as f:
                    f.write('Experiment     : ' + self.var['theSample'] + '\n')
                    f.write('Experiment type: ' + self.var['theMeasurementType'] + '\n')
                    f.write(self.var['theDate'] + '\n\n')
                    f.write('Laser: ' + self.var['Laser'] + '\n')
                    f.write('Polarizator steps: ' + str(self.var['NPSteps']) + '\n')
                    f.write('Analizator steps: ' + str(self.var['NASteps']) + '\n')
                    f.write('Images averaged: ' + str(self.var['Nimages']) + '\n')
                    f.write('Dark images averaged: ' + str(self.var['NdarkImages']) + '\n\n')
                    f.close()
            except IOError:
                print('EE problem opening ' + self.statFName +'!')
        
        print("done", flush=True)

        print("Reading Camera variables..", end='', flush=True)
        config = configparser.ConfigParser()
        config.read(self.variablesFName)
        if config.has_section("Camera"):
            self.var['CameraModel'] = config.get('Camera', 'CameraModel')
            self.var['imgw'] = config.getint('Camera', 'imageWidth')
            self.var['imgh'] = config.getint('Camera', 'imageHeight')
            self.var['theColor'] = config.get('Camera', 'ColorChannel')
            self.var['ExposureTime'] = config.getfloat('Camera', 'ExposureTime')
        else:
            filesList=glob.glob(os.path.join(self.dataDir, 'raw_images', '') + "*." + self.raw_extension)
            if len(filesList) > 0:   # if there are raw files
                filesList.sort(key=os.path.getmtime)
                #self._writeVarToFile('\n\n[Camera]')

                # Checking image parameters
                cmd="exiftool -s -s -s -Model " + filesList[0]
                self.var['CameraModel'] = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()[0].decode("utf-8").rstrip()

                cmd="exiftool -s -s -s -ExposureTime " + filesList[0]
                tmpstr = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()[0].decode("utf-8")
                if len(tmpstr) > 0:
                    self.var['ExposureTime'] = float(tmpstr.rstrip())
                else:
                    print('EE got "' + tmpstr +'" instead of ExposureTime from file ' + filesList[0])  
                    sys.exit()

                cmd="/usr/bin/4channels " + str(filesList[0]) + " >/dev/null 2>&1"
                os.system(cmd)
                fname = filesList[0]+"."+"R"+".tiff"
                self.var['imgw'], self.var['imgh'] = 0, 0
                self.var['imgw'], self.var['imgh'] = Image.open(fname).size

                # Checking the channel we use
                cmd="identify -format \"%[mean]\" " + filesList[0]+"."+"R"+".tiff"
                mR = float(subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()[0])
                cmd="identify -format \"%[mean]\" " + filesList[0]+"."+"G"+".tiff"
                mG = float(subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()[0])
                cmd="identify -format \"%[mean]\" " + filesList[0]+"."+"B"+".tiff"
                mB = float(subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()[0])
                themean={'R':mR,'G':mG,'B':mB}
                self.var['theColor']=max(themean, key=themean.get)

                # write images info to file
                try:
                    with open(self.statFName, "a") as f:
                        f.write("Camera model: " + self.var['CameraModel'] + '\n')
                        f.write('ExposureTime: ' + str(self.var['ExposureTime']) + '\n')
                        f.write('Single-channel image size: ' + str(self.var['imgw']) + 'x' + str(self.var['imgh']) + '\n\n')
                        f.close()
                except IOError:
                    print('EE problem opening ' + self.statFName +'!')

                # remove unnecessary tiff files
                for f in glob.glob(os.path.join(self.dataDir, 'raw_images', '*.tiff')):
                    os.remove(f)
        print("done") # reading camera variables


    def _unpRAW(self, fName):
        # extract the image from RAW file
        cmd="/usr/bin/4channels -B " + fName + " >/dev/null 2>&1"
        os.system(cmd)
        if os.path.isfile(fName + "." + self.var['theColor'] + ".tiff"):
            os.rename(fName + "." + self.var['theColor'] + ".tiff", fName + "." + self.var['theColor'] + ".tif")
            os.remove(fName) # remove the raw file
        for f in glob.glob(fName + "*.tiff"):
            os.remove(f)  # remove all .tiff files
        print(".", end='', flush=True)


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
            arr = np.zeros((self.var['imgh'], self.var['imgw']), dtype=np.float32)
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
            self.var['imageNoise'] = 3*np.std(darkimg)

            # write dark image info to file
            # using the last image which is still in darkimg
            try:
                with open(self.statFName, "a") as f:
                    f.write('The dark image statistics:\n')
                    f.write("  Average: {:8.3f}\n".format(np.average(darkimg)))
                    f.write("  Median:  {:8.3f}\n".format(np.median(darkimg)))
                    f.write("  stdev:   {:8.5f}\n".format(np.std(darkimg)))
                    f.write("  Noise:   {:8.5f}\n\n".format(self.var['imageNoise']))
                    f.close()
            except IOError:
                print('EE problem opening ' + self.statFName +'!')

            # save the dark file
            np.save("dark.npy", darkimg)
            
            if self.write16bit:
                if not os.path.isdir("16bit"):
                    cmd = "mkdir 16bit >/dev/null 2>&1"
                    os.system(cmd)
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
        filesList=glob.glob(os.path.join(self.dataDir, 'Images', '???_???.npy'))
        if len(filesList) != self.var['NASteps']*self.var['NPSteps']:
            print('Assembling images:')
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
            for i in range(self.var['NPSteps']):
                for j in range(self.var['NASteps']):
                    filesInAngle=filesList[(i*self.var['NASteps'] +j)*self.var['Nimages']:(i*self.var['NASteps'] +j)*self.var['Nimages']+self.var['Nimages']]
                    newFName = str(format(i+1, '03d')) + "_" + str(format(j+1, '03d'))                    
                    """
                    print('\n\n** i=' + str(i) + ' j=' + str(j))
                    print('** files in angle:', len(filesInAngle))
                    print('** files:', filesInAngle)
                    print('** files from:', (i*self.var['NASteps'] +j)*self.var['Nimages'])
                    print('** files to  :', (i*self.var['NASteps'] +j)*self.var['Nimages']+self.var['Nimages'])                
                    print('** out file:', newFName)
                    """
                    arr = np.zeros((self.var['imgh'], self.var['imgw']), dtype=np.float32)
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
            if len(glob.glob(os.path.join(theDir, '*.npy'))) == self.var['NPSteps']*self.var['NASteps']:
                for f in glob.glob(os.path.join(os.getcwd(), 'raw_images', '*.tif')):
                    os.remove(f)
                os.removedirs(os.path.join(os.getcwd(), 'raw_images'))


    def _findTheCenter(self):
        if not os.path.isfile(self.varFName):
        # find the center of the bright spot in the image
        # and save it to variables
            config = configparser.ConfigParser()
            config.read(self.variablesFName)
            print("Reading the image constraints...", end="")
            # try to read variables
            if config.has_option('Camera', 'xmin'):
                self.var['xcenter'] = config.getint('Camera', 'xcenter')
                self.var['ycenter'] = config.getint('Camera', 'ycenter')
                self.var['xmin'] = config.getint('Camera', 'xmin')
                self.var['xmax'] = config.getint('Camera', 'xmax')
                self.var['ymin'] = config.getint('Camera', 'ymin')
                self.var['ymax'] = config.getint('Camera', 'ymax')
                self.var['circlerad'] = config.getint('Camera', 'circlerad')
            else:
                # find the variables and save them to file
                filesList=glob.glob(os.path.join(self.dataDir, 'Images', '???_???.npy'))
                if len(filesList) > 0:
                    arr = np.load(filesList[0])
                    themin = np.min(arr)
                    themax = np.max(arr)
                    theRange = themax - themin
                    threshh = 0.01*themax
                    thecol = arr.sum(axis=0)/self.var['imgh']          # sum columns
                    self.var['xmin'] = np.min(np.where(thecol > threshh))
                    self.var['xmax'] = np.max(np.where(thecol > threshh))
                    self.var['xcenter'] = self.var['xmin'] + round((self.var['xmax'] - self.var['xmin'])/2)
                    print(".", end="")
                    #self._writeVarToFile("\nxmin=" + str(self.var['xmin']))
                    #self._writeVarToFile("xmax=" + str(self.var['xmax']))
                    therow = arr.sum(axis=1)/self.var['imgw']          # sum rows
                    self.var['ymin'] = np.min(np.where(therow > threshh))
                    self.var['ymax'] = np.max(np.where(therow > threshh))
                    #self._writeVarToFile("ymin=" + str(self.var['ymin']))
                    #self._writeVarToFile("ymax=" + str(self.var['ymax']))
                    self.var['ycenter'] = self.var['ymin'] + round((self.var['ymax'] - self.var['ymin'])/2)
                    print(".", end="")
                    #self._writeVarToFile("xcenter=" + str(self.var['xcenter']))
                    #self._writeVarToFile("ycenter=" + str(self.var['ycenter']))
                    self.var['circlerad'] = round((self.var['ymax'] - self.var['ymin'])/2) - self.decrRadius
                    #self._writeVarToFile("circlerad=" + str(self.var['circlerad']))
            print("done")


    def _blurImg(self, theNumpyImgFName):
        # Blur single numpy image from Images/ using gaussian filter
        # and put the result to Images_smoothed/
        arr = sp.ndimage.filters.gaussian_filter(np.load(theNumpyImgFName), self.var['theSigma'])
        np.save(os.path.join(os.path.join(self.dataDir, 'Images_smoothed', os.path.basename(theNumpyImgFName))), arr)
        print('.', end='', flush=True)


    def _blurAllImages(self):
        # Blur all the images
        if self.var['theSigma'] > 0:
            # get the list of files
            filesList=glob.glob(os.path.join(self.dataDir, 'Images', '???_???.npy'))
            filesList.sort(key=os.path.getmtime)

            # check if there are images in the Images dir
            if len(filesList) == 0:
                print("EE There are no images in the directory:", len(filesList))
                print("Aborting...")
                sys.exit()

            # check if number of files equals variables
            if len(filesList) != self.var['NPSteps']*self.var['NASteps']:
                print("EE Total number of images expected:", self.var['NPSteps']*self.var['NASteps'])
                print("EE Images in the directory:", len(filesList))
                print("Mismatch! Aborting...")
                sys.exit()

            # Blur images using gaussian filter
            # make the directory for images if it does not exists
            thesDir = os.path.join(self.dataDir, 'Images_smoothed')
            if not os.path.isdir(thesDir):
                cmd="mkdir " + thesDir + " >/dev/null 2>&1"
                os.system(cmd)

            print("    Blurring images with sigma=" + str(self.var['theSigma']) + " ..", end='', flush=True)
            pool = multiprocessing.Pool(self.NThreads)
            pool.map(self._blurImg, filesList)
            pool.close()
            pool.join()
            print("done")  # Blur images          


    def _mkCharacterization(self):
        # process polaroids characterization
        theTime = 0.

        print("    Loading data..", end='', flush=True)
        if self.var['theSigma'] > 0:   # Load blurred images to memory             
            filesList=glob.glob(os.path.join(self.dataDir, 'Images_smoothed', '???_???.npy'))
        else:
            # Load images to memory
            filesList=glob.glob(os.path.join(self.dataDir, 'Images', '???_???.npy'))

        filesList.sort(key=os.path.getmtime)            
        allImgArr = np.array([np.load(fname) for fname in filesList])
        print('.done')

        n_z, n_y, n_x = allImgArr.shape   # get dimensions of the array
        print("    Input array shape: " + str(n_x) + "x" + str(n_y) + ", images: " + str(n_z))

        # Load calibrated values of x variable:
        if os.path.isfile(self.xarrayFName):
            xarray = np.loadtxt(self.xarrayFName)
        else:
            print('WW no APosCalibrated.dat file found! Uncalibrated data will be used instead')
            xarray=np.arange(-(n_z-1)/2, (n_z-1)/2+1, 1)*self.AStepRad

        # initialize arrays for variables
        thedeg = np.zeros(shape=(n_y, n_x), dtype=np.float32)
        extRatio = np.zeros(shape=(n_y, n_x), dtype=np.float32)
        residuals = np.zeros(shape=(n_y, n_x), dtype=np.float32)
        thedeg[:], extRatio[:], residuals[:] = np.nan, np.nan, np.nan
        a = np.empty(shape=(n_y, n_x), dtype=np.float32)
        b = np.empty(shape=(n_y, n_x), dtype=np.float32)
        c = np.empty(shape=(n_y, n_x), dtype=np.float32)
        a[:], b[:], c[:] = np.nan, np.nan, np.nan
        # Make array of weights
        wyArray = np.zeros(shape=(n_z, n_y, n_x), dtype=np.float32)
        # just to avoid zero division:
        allImgArr[allImgArr == 0.] = 0.0000000000001
        wyArray = 1 / np.absolute(allImgArr)

        # Calculate how much memory taken by the data
        # *********
        dataSize = str(round((allImgArr.nbytes * 8)/ 1024 / 1024,2))
        print("    Data size in memory: " + dataSize + " Mb")

        global fitd
        def fitd(dy, dx):
        # function which fits the single point of data
            if (dx - self.var['xcenter']) ** 2 + (dy - self.var['ycenter']) ** 2 < self.var['circlerad'] ** 2:
                sol = np.polyfit(xarray, allImgArr[:, dy, dx], 2, rcond=0.00001, full=True, w=wyArray[:, dy, dx])
                return np.append(sol[0], sol[1][0])
            else:
                return np.nan, np.nan, np.nan, np.nan

        print("    Fitting parabolas --- PLEASE WAIT UNTIL IT FINISH THE TASK  ---", end='', flush=True)
        theStamp = time.time()

        pool = multiprocessing.Pool(self.NThreads)
        results = pool.starmap(fitd, ((dy, dx) for dy in range(n_y) for dx in range(n_x)))
        pool.close()
        pool.join()

        newarr = np.asarray(results, dtype=np.float32).reshape(n_y, n_x, 4)
        a = newarr[:, :, 0]
        b = newarr[:, :, 1]
        c = newarr[:, :, 2]
        residuals = newarr[:, :, 3]

        theTime = time.time() - theStamp

        # --- parameters calculation ---
        # just to avoid zero division:
        a[a == 0.] = 0.0000000000001

        # ... angle
        thedeg = -(360/(2*pi))*b/(2*a)

        # Extinction ratio
        extRatio = (c/a - (b / (2 * a)) ** 2) / 2

        # save the calculated images
        np.save("min_angle", thedeg)
        np.save("extRatio", extRatio)
        np.save("residuals", residuals)
        np.save("a", a)
        np.save("b", b)
        np.save("c", c)

        if self.write16bit:
            im = Image.fromarray(np.uint16(np.around(extRatio)), mode='I;16')
            im.save("16bit/extRatio.tiff", "TIFF")
            im = Image.fromarray(np.uint16(np.around(thedeg)), mode='I;16')
            im.save("16bit/min_angle.tiff", "TIFF")
            im = Image.fromarray(np.uint16(np.around(residuals)), mode='I;16')
            im.save("16bit/residuals.tiff", "TIFF")

        #print('One parabola fitting took {:.6f} ms'.format(theTime*1000/theCount))
        print(' The fitting took {:.2f} s'.format(theTime))
        #print(str(theCount) + ' parabolas has been fitted')



    def show2d(self, imgFName):
        #fname=sys.argv[1]
        extension = os.path.splitext(imgFName)[1]

        # open the image to show
        if extension == ".tif":
            arr=np.array(Image.open(imgFName))
        if extension == ".tiff":
            arr=np.array(Image.open(imgFName))
        elif extension == ".npy":
            arr = np.load(imgFName)

        #print("image: ", imgFName)
        #print("ext: ", extension)
        #print("array type:", arr.dtype)

        fig = plt.figure()
        plt.margins(0)
        plt.title(imgFName)
        plt.tight_layout(pad=0)
        plt.imshow(arr, cmap='gray', interpolation = 'none')
        #plt.savefig(imgFName,dpi=my_dpi*2,bbox_inches='tight')
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
