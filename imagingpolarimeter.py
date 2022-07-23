""" Imaging polarimeter module for interpreting the obtained data

"""


from cmath import pi
import sys
import glob
import os
import math
from PIL import Image
#import numba as nb
import numpy as np
from numpy import sqrt
#from numpy import savez_compressed
import scipy as sp
#import scipy.ndimage
from scipy import ndimage
import subprocess
from matplotlib import pyplot as plt
sys.path.append(os.getcwd())


__author__ = "Serhiy Kobyakov"
__authors__ = ["Serhiy Kobyakov"]
__copyright__ = "Copyright 2022, Serhiy Kobyakov"
__credits__ = ["Yaroslav Shopa"]
__date__ = "2022/03/12"
__deprecated__ = False
__email__ =  "skobyakov@uksw.edu.pl"
__license__ = "MIT"
__maintainer__ = "Serhiy Kobyakov"
__status__ = "Production"
__version__ = "0.0.1"



# if no variables.m file - quit
if not os.path.isfile(os.path.join(os.getcwd(), 'variables.m')):
    print("No variables.m file found!\nAborting...")
    sys.exit()


# retrieve all the experiment variables
# left it here, almost all the functions need it
with open(os.path.join(os.getcwd(), 'variables.m')) as f: exec(f.read())


# x array positions are in the file:
global xarrayFName
xarrayFName = 'APosCalibrated.dat'



def add_to_variables(str):
# save the variables to the file "variables.m"
    try:
        with open("variables.m", "a") as f:
            f.write(str + '\n')
            f.close()
    except IOError:
        print('problem opening variables.m!')    


def _getvariables(extension):
# get basic info about images
    print("Getting info about images:")
    filesList=glob.glob(os.path.join(os.getcwd(), 'raw_images', '') + "*." + extension)

    process = subprocess.Popen(['exiftool', '-s', '-s', '-s', '-ExposureTime', '-n', filesList[0]],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    #print(stdout.decode("utf-8").strip())
    global ExposureTime
    ExposureTime = float(stdout.decode("utf-8").strip())
    #print(ExposureTime)
    add_to_variables("ExposureTime=" + str(ExposureTime) + "   # ExposureTime of the images in the experiment")

        # get image size
    cmd="/usr/bin/4channels " + str(filesList[0]) + " >/dev/null 2>&1"
    os.system(cmd)
    fname=filesList[0]+"."+"R"+".tiff"
    global imgh, imgw    
    imgw,imgh=Image.open(fname).size
    add_to_variables("imgw=" + str(imgw) + "      # image width")
    add_to_variables("imgh=" + str(imgh) + "      # image height")
    print("    output image width=", imgw)
    print("    output image height=", imgh)

    # Find out what channel we'll use
    global theColor
    cmd="identify -format \"%[mean]\" " + filesList[0]+"."+"R"+".tiff"
    mR = float(subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()[0])
    cmd="identify -format \"%[mean]\" " + filesList[0]+"."+"G"+".tiff"
    mG = float(subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()[0])
    cmd="identify -format \"%[mean]\" " + filesList[0]+"."+"B"+".tiff"
    mB = float(subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()[0])
    themean={'R':mR,'G':mG,'B':mB}

    #theColor=max(themean, key=themean.get)
    theColor = "R"

    print('    The most intense channel is:', theColor)
    add_to_variables("theColor=\"" + theColor + "\"   # the most intense channel of the first image")

    # remove unnecessary tiff files
    for f in glob.glob(os.path.join(os.getcwd(), '') + "/raw_images/*.tiff"):
        os.remove(f)  # remove all .tiff files
    #print('')

    if os.path.isfile(xarrayFName):
      #  xarr = np.zeros(lines, np.float16)
        global xarray
        xarray = np.loadtxt(xarrayFName)
        print('    xarray size:', xarray.size)
    else:
        print('no APosCalibrated.dat file found!')    


def _raw2tif(intheDir, extension):
# unpack raw files and get the linear images we need
    theColor="R"    
    filesList=glob.glob(os.path.join(intheDir, '') + "*." + extension)
    filesList.sort(key=os.path.getmtime)
    #tfilesList=glob.glob(os.path.join(intheDir, '') + "*.tif")
    if os.path.isdir(intheDir) and len(filesList) > 0:
        print('Unpack raw images from ' + intheDir + ':')
        print("    Converting images.", end='', flush=True)
        for j in range(len(filesList)):
        # pull out the channels from raw image
            cmd="/usr/bin/4channels -B " + str(filesList[j]) + " >/dev/null 2>&1"
            os.system(cmd)
            if os.path.isfile(filesList[j] + "." + theColor + ".tiff"):
                os.rename(filesList[j] + "." + theColor + ".tiff", filesList[j] + "." + theColor + ".tif")
                os.remove(filesList[j]) # remove the raw file
            # remove unnecessary tiff files
            for f in glob.glob(os.path.join(intheDir, '') + "*.tiff"):
                os.remove(f)  # remove all .tiff files
            print(".", end='', flush=True)          
        print("done")  # unpacking raw files
    print('')


def _backupCR2():
# backup CR2 files
    theDir = os.path.join(os.getcwd(), '') + "backup"
    os.mkdir(theDir)
    if os.path.isdir(theDir):
        os.system("mv *.CR2 " + theDir +" >/dev/null 2>&1")
        os.system("cp variables.m " + theDir +" >/dev/null 2>&1")
        os.system("cp log.txt " + theDir +" >/dev/null 2>&1")
    theDir = os.path.join(os.getcwd(), '') + "backup/background_images"        
    os.mkdir(theDir)
    if os.path.isdir(theDir):
        cmd="mv " + os.path.join(os.getcwd(), '') + "background_images/*.CR2 " + theDir + " >/dev/null 2>&1"
        os.system(cmd)


def _mkdark():
# assemble the dark image
    if not os.path.isfile('dark.tiff') and os.path.isdir(os.path.join(os.getcwd(), 'raw_background_images')):
        print('Making dark image:')
        # get list of tif files in the bg directory
        bgfilesList=glob.glob(os.path.join(os.getcwd(), 'raw_background_images','*.tif'))
        bgfilesList.sort(key=os.path.getmtime)
        
        # read the dark images to array
        print("    Reading " + str(len(bgfilesList)) +" dark images into array..", end='', flush=True)
        darkarr=np.array([np.array(Image.open(fname)) for fname in bgfilesList])
        print(".done")
        n_z, n_y, n_x = darkarr.shape   # get dimensions of the array
        print("    The array shape: x=", n_x, " y=", n_y, "z=", n_z, " size in memory: "+ str(round(darkarr.nbytes / 1024 / 1024,2)) + " Mb")
        #print("dark images array size in memory: " + str(round(darkarr.nbytes / 1024 / 1024,2)) + " Mb")

        # open file for statistical information about background images
        arr = np.zeros((imgh, imgw), np.float32)
        arr = np.array(Image.open(bgfilesList[0]),dtype=np.float32)
        logImgQuality(arr, "dark.stat.txt", "Single image from the set of " + str(len(bgfilesList)) +" images:", False)

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

        print("    Averaging " + str(len(bgfilesList)) +" dark images into singe image..", end='', flush=True)
        darkimg=np.zeros((n_y,n_x), dtype=np.float32)
        for z in range(n_z):
            darkimg = darkimg + darkarr[z, 0:n_y, 0:n_x]
        darkimg=darkimg/n_z

        # write statistics to file
        logImgQuality(darkimg, "dark.stat.txt", "Averaged and filtered dark image:", False)

        # save the dark image file
        im = Image.fromarray(darkimg, mode='F')
        im.save("dark.tiff", "TIFF")  # save the dark image

        if os.path.isfile('dark.tiff'):
            for f in glob.glob(os.path.join(os.getcwd(), 'raw_background_images','*.tif')):
                os.remove(f)
            os.removedirs(os.path.join(os.getcwd(), 'raw_background_images'))

        print(".done")  # working on the dark image


def _maketheimages():
# make the images for the experiment
    theDir = os.path.join(os.getcwd(), 'images')
    if not os.path.isdir(theDir):
        cmd="mkdir " + theDir + " >/dev/null 2>&1"
        os.system(cmd)

    filesList=glob.glob(theDir + "/???_???.tiff")
    if len(filesList) != NASteps*NPSteps:
        print('\nAssembling images:')
        # open the dark image
        if os.path.isfile("dark.tiff"):
            dark = np.array(Image.open("dark.tiff"), dtype=np.float32)
        else:
            print("No dark.tiff found!")
            sys.exit()

        print('** dark aray size:', dark.size)

        theDir = os.path.join(os.getcwd(), '') + "images/"
        if not os.path.isdir(theDir):
            cmd="mkdir " + theDir + " >/dev/null 2>&1"
            os.system(cmd)

        filesList=glob.glob(os.path.join(os.getcwd(), '') + "raw_images/*.tif")
        filesList.sort(key=os.path.getmtime)

        # get noise info and other statistics for the first image
        arr = np.zeros((imgh, imgw), np.float32)
        arr = np.array(Image.open(filesList[0]), dtype=np.float32)
        logImgQuality(arr, "image.stat.txt", "Single image from the set of " + str(len(filesList)) +" images:", False)

        # make images for each Analizator position
        print("    Making images for each Analizator position.", end='', flush=True)
        for i in range(NPSteps):
            for j in range(NASteps):
                filesInAngle=filesList[(i*NASteps +j)*Nimages:(i*NASteps +j)*Nimages+Nimages]
                outFileName = theDir + str(format(i+1, '03d')) + "_" + str(format(j+1, '03d')) + ".tiff" # get the final file name for this angle
                
                print('\n\n** i=' + str(i) + ' j=' + str(j))
                print('** files in angle:', len(filesInAngle))
                print('** files:', filesInAngle)
                print('** files from:', (i*NASteps +j)*Nimages)
                print('** files to  :', (i*NASteps +j)*Nimages+Nimages)                
                print('** out file:', outFileName)
                
                arr = np.zeros((imgh, imgw), np.float32)
                for n in range(len(filesInAngle)):
                    fname=filesInAngle[n]  #+"."+theColor+".tiff"
                    #print("\nfname=",fname)
                    arr = arr + np.array(Image.open(fname), dtype=np.float32)
                    print('.', end='', flush=True)
                arr = arr/len(filesInAngle) - dark
                #arr = arr - dark
                im = Image.fromarray(arr, mode='F')
                im.save(outFileName, "TIFF")  # save image
                print('.', end='', flush=True)        
        print("done")

        # use the last image processed which is still in the arr array
        logImgQuality(arr, "image.stat.txt", 'Final image:', False)

        #theFilesList=glob.glob(os.path.join(os.getcwd(), 'images','*.tiff'))
        if len(glob.glob(os.path.join(os.getcwd(), 'images','*.tiff'))) == NPSteps*NASteps:
            for f in glob.glob(os.path.join(os.getcwd(), 'raw_images','*.tif')):
                os.remove(f)
            os.removedirs(os.path.join(os.getcwd(), 'raw_images'))

        print('')


def _maketheavgimages():
# make the images for the experiment
    theDir = os.path.join(os.getcwd(), 'images')
    if not os.path.isdir(theDir):
        cmd="mkdir " + theDir + " >/dev/null 2>&1"
        os.system(cmd)

    filesList=glob.glob(theDir + "/???_???.tiff")
    if len(filesList) != NASteps*NPSteps:
        print('Assembling images:')
        # open the dark image
        if os.path.isfile("dark.tiff"):
            dark = np.array(Image.open("dark.tiff"), dtype=np.float32)
        else:
            print("No dark.tiff found!")
            sys.exit()

        theDir = os.path.join(os.getcwd(), '') + "images/"
        if not os.path.isdir(theDir):
            cmd="mkdir " + theDir + " >/dev/null 2>&1"
            os.system(cmd)

        filesList=glob.glob(os.path.join(os.getcwd(), '') + "raw_images/*.tif")
        filesList.sort(key=os.path.getmtime)

        # get noise info and other statistics for the first image
        arr = np.zeros((imgh, imgw), np.float32)
        arr = np.array(Image.open(filesList[0]),dtype=np.float32)
        logImgQuality(arr, "image.stat.txt", "Single image from the set of " + str(len(filesList)) +" images:", False)

        # make images for each Analizator position
        print("    Making images for each Analizator position.", end='', flush=True)
        #for i in range(NPSteps):

        for j in range(NASteps):
            arr = np.zeros((imgh, imgw), np.float32)
            outFileName = theDir + str(format(1, '03d')) + "_" + str(format(j+1, '03d')) + ".tiff" # get the final file name for this angle
            #print(outFileName)
            arr = np.zeros((imgh, imgw), np.float32)
            for n in range(Nimages):
                arr = arr + np.array(Image.open(filesList[j + n*NASteps]),dtype=np.float32) - dark
            arr = arr/Nimages
            #if j % math.ceil(NPSteps*NASteps/100):
            print('.', end='', flush=True)

            # save image
            im = Image.fromarray(arr, mode='F')
            im.save(outFileName, "TIFF")  # save image
            print('.', end='', flush=True)        
        print("done")

        # use the last image processed which is still in the arr array
        logImgQuality(arr, "image.stat.txt", 'Final image:', False)

        if len(glob.glob(os.path.join(os.getcwd(), 'images','*.tiff'))) == NPSteps*NASteps:
            for f in glob.glob(os.path.join(os.getcwd(), 'raw_images','*.tif')):
                os.remove(f)
            os.removedirs(os.path.join(os.getcwd(), 'raw_images'))

        print('')



def logImgQuality(arr, thefile, theStr, saveNoiseVal):
# log the image quality - noise
    fstat = open(thefile, "a")

    subarr = arr[:imgh, imgw-150:]  # slice a part of image

    theSigma=np.std(subarr)
    theoldSigma=3*theSigma
    theMedian=np.median(subarr)
    while (theoldSigma - theSigma)/theSigma > 1e-10:
        theoldSigma=theSigma
        subarr[abs(subarr - theMedian) > 3*theSigma] = theMedian    
        theSigma=np.std(subarr)
        theMedian=np.median(subarr)

    fstat.write("\n" + theStr + "\n")
    fstat.write("  Average: {:8.3f}\n".format(np.average(subarr)))
    fstat.write("  Median:  {:8.3f}\n".format(np.median(subarr)))
    fstat.write("  stdev:   {:8.5f}\n".format(np.std(subarr)))
    fstat.write("  Noise:   {:8.5f}\n".format(3*np.std(subarr)))
    fstat.close

    if saveNoiseVal:
        add_to_variables("ImageNoise={:8.5f}\n".format(3*np.std(subarr)))


def _getDX():
# get the dx shift between camera image and the final quadratic images
    # get list of files
    theDir = os.path.join(os.getcwd(), '') + "images/"
    filesList=glob.glob(theDir + "???_???.tiff")

    # open first image
    arr = np.zeros((imgh, imgw), np.float32)
    arr = np.array(Image.open(filesList[0]),dtype=np.float32)
    arr[arr > 40] = 40.

    theY, theX = ndimage.center_of_mass(arr)
    #print("x center =", theX)
    global theDX
    theDX = round(theX) - round(imgh/2)
    #print("shift = ", theDX)
    add_to_variables("theDX=" + str(theDX) + "   # the shift between camera image and the final quadratic images")
    return theDX



def _mker(theDX, fromTop, theSigma):
    print('Making extension ratio and angle images:')
    # max saturation level for image sensor
    maxI = 1000.0


    # get list of files
    theDir = os.path.join(os.getcwd(), 'images')
    filesList=glob.glob(os.path.join(theDir, "???_???.tiff"))
    filesList.sort(key=os.path.getmtime)

    #check if number of files equals variables
    if len(filesList) != NPSteps*NASteps:
        print("Total number of images expected:", AllImages)
        print("Images in the directory:", len(filesList))
        print("Mismatch! Aborting...")
        sys.exit()

    # get all images to array
    print("    Loading data..", end='', flush=True)
    allImgArr = np.array([np.array(Image.open(fname)) for fname in filesList])
    #savez_compressed('allImgArr.npz', allImgArr)    # save all images to file
    print('.done')

    n_z, n_y, n_x = allImgArr.shape   # get dimensions of the array
    print("    Array shape: x=", n_x, " y=", n_y, "images:", n_z)
    print("    Data size in memory: " + str(round(allImgArr.nbytes / 1024 / 1024,2)) + " Mb")

    thesDir = os.path.join(os.getcwd(), 'images_smoothed')
    if not os.path.isdir(thesDir):
        cmd="mkdir " + thesDir + " >/dev/null 2>&1"
        os.system(cmd)

    # Blur image using gaussian filter
    print("    Blurring the data with sigma=" + str(theSigma) + " ..", end='', flush=True)
    if theSigma > 0:
        for dz in range(n_z):
            arr = sp.ndimage.filters.gaussian_filter(allImgArr[dz, :, :], theSigma)
            allImgArr[dz, :, :] = arr
            im = Image.fromarray(arr, mode='F')
            im.save(os.path.join(thesDir, os.path.basename(filesList[dz])), "TIFF")  # save image
            print('.', end='', flush=True)
    logImgQuality(arr, 'image.stat.txt', 'Final image (smoothed with sigma='+str(theSigma)+'):', True)
    print('.done')

    add_to_variables("fromTop=" + str(fromTop) + "   # the additional shift between top and left coordiantes which adds to the final quadratic images")

    xmin = theDX + fromTop
    xmax = theDX + imgh - fromTop
    ymin = fromTop
    ymax = imgh - fromTop
    side = imgh - 2*fromTop

    print("    The array shape: " + str(xmax - xmin) + "x" + str(ymax - ymin))

    # array for parabola's minima
    minparr=np.zeros(shape=(n_y, n_x), dtype=np.float32)
    # array for parabola's minima
    allSigmaI=np.zeros(shape=(n_y, n_x), dtype=np.float32)

    # calculate x range
    #xarray=np.arange(-(n_z-1)/2, (n_z-1)/2+1, 1)*AStepRad
    #xarray = np.loadtxt(fname)

    if os.path.isfile(xarrayFName):
      #  xarr = np.zeros(lines, np.float16)
        #global xarray
        xarray = np.loadtxt(xarrayFName)
        #print(xarray)
    else:
        print('no APosCalibrated.dat file found!') 

    print("    Fitting parabolas..", end='', flush=True)
    # initialize arrays for variables
    a=np.zeros(shape=(n_y, n_x), dtype=np.float32)
    b=np.zeros(shape=(n_y, n_x), dtype=np.float32)
    c=np.zeros(shape=(n_y, n_x), dtype=np.float32)    
#    thedeg=np.zeros(shape=(n_y, n_x), dtype=np.float32)
#    er=np.zeros(shape=(n_y, n_x), dtype=np.float32)
    thedeg=np.zeros(shape=(side, side), dtype=np.float32)
    er=np.zeros(shape=(side, side), dtype=np.float32)
    allVariance=np.zeros(shape=(side, side), dtype=np.float32)
    #wormgeardx=np.zeros(shape=(NASteps), dtype=np.float32)
    #wgcount = 0

    for dx in range(xmin, xmax):
        print('.', end='', flush=True)
        for dy in range(ymin, ymax):
            #if math.sqrt((dx-n_x/2)*(dx-n_x/2) + (dy-n_y/2)*(dy-n_y/2)) < n_y/2:
                #yarrr=allImgArr[0:n_z, dy, dx]
                yarr=allImgArr[:, dy, dx]
                xarr=xarray

                # discard data that are too close to the sides
                for j in reversed(range(n_z)):
                    index = j-(NASteps-1)/2
                    if abs(index*math.sin(index/2)) > (NASteps-1)/10:
                        yarr = np.delete(yarr, j)
                        xarr = np.delete(xarr, j)                

                #yarr=yarrr
                # remove saturated values
                arrsize = yarr.shape
                for j in reversed(range(arrsize[0])):
                    if yarr[j] > maxI:
                        yarr = np.delete(yarr, j)
                        xarr = np.delete(xarr, j)
                        #yarr[j] = np.nan
                        #xarr[j] = np.nan


                # fit the data
                a[dy, dx], b[dy, dx], c[dy, dx]=np.polyfit(xarr, yarr, 2)
                thedeg[dy-ymin, dx-xmin] = -(360/(2*pi))*b[dy, dx]/(2*a[dy, dx])
                er[dy-ymin, dx-xmin] = c[dy, dx] - b[dy, dx] ** 2 /(4 * a[dy, dx])
                # Extinction ratio = I crossed / I aligned
                # er = (I / exp time) / (3000 *4000 / 1 ) = I/ (ExposureTime * 12000000)
                er[dy-ymin, dx-xmin] = er[dy-ymin, dx-xmin] / (ExposureTime * 12000000)

                # calculate variance per NASteps
                variance=0
                for x,y in zip(xarr, yarr):
                    variance=variance + (y - (a[dy, dx]*x**2 + b[dy, dx]*x + c[dy, dx])) ** 2 #(y - (a[dy, dx]*x**2 + b[dy, dx]*x + c[dy, dx]))
                if variance > 0:
                    variance=math.sqrt(variance)/NASteps
                else:
                    variance=0
                allVariance[dy-ymin, dx-xmin] = variance
    
    # save the calculated images
    im = Image.fromarray(thedeg, mode='F')
    im.save("min_angle.tiff", "TIFF")  # save image
    im = Image.fromarray(er, mode='F')
    im.save("er.tiff", "TIFF")  # save image
    im = Image.fromarray(allVariance, mode='F')
    #savez_compressed('allVariance.npz', allVariance)
    im.save("variance.tiff", "TIFF")  # save image

    
    print("done")
    print()



def _mkdata45(theDX, fromTop):
    print('Calculating the data for 45 degree scan:')
    # max saturation level for image sensor
    #maxI = 1000.0


    # get list of files
    theDir = os.path.join(os.getcwd(), 'images')
    filesList=glob.glob(os.path.join(theDir, "???_???.tiff"))
    filesList.sort(key=os.path.getmtime)

    #check if number of files equals variables
    if len(filesList) != NPSteps*NASteps:
        print("Total number of images expected:", AllImages)
        print("Images in the directory:", len(filesList))
        print("Mismatch! Aborting...")
        sys.exit()


    # calculate x range
    xarr=np.arange(-(NASteps-1)/2, (NASteps-1)/2+1, 1)
    xarr=xarr*AStepRad

    #add_to_variables("fromTop=" + str(fromTop) + "   # the additional shift between top and left coordiantes which adds to the final quadratic images")

    xmin = theDX + fromTop
    xmax = theDX + imgh - fromTop
    ymin = fromTop
    ymax = imgh - fromTop
    side = imgh - 2*fromTop

    #xarray=np.arange(-(NASteps-1)/2, (NASteps-1)/2+1, 1)*AStepRad

    if os.path.isfile(xarrayFName):
      #  xarr = np.zeros(lines, np.float16)
        #global xarray
        xarray = np.loadtxt(xarrayFName)
        #print(xarray)
    else:
        print('no APosCalibrated.dat file found!')     

    print("    Calculating the data:", end='', flush=True)
    
    f = open('log.txt', "r")
    lines = f.readlines()
    f.close

    fwg = open('wormgeardx.dat', "w")
    wormgeardx=np.zeros(shape=(NASteps), dtype=np.float32)
    for dz in range(NASteps-1):
        imgarr = np.array(Image.open(filesList[dz]),dtype=np.float32)
        arr = imgarr[ymin:ymax, xmin:xmax]
        wormgeardx[dz] = np.average(arr)
        #print(xarray[dz], wormgeardx[dz])
        if dz % 10:
            print('.', end='', flush=True)
        #thetokens = lines[dz+5].split("\t")
        #fwg.write(lines[dz+5].split("\t")[5] + "\t" + str(xarray[dz]) + "\t" + str(wormgeardx[dz]) + "\n")
        fwg.write(lines[dz+4].split("\t")[5].rstrip() + "\t" + str(wormgeardx[dz]) + "\n")
    fwg.close
    

    print("done")      
    print()




def _showp(dx, dy):
    # max saturation level for image sensor
    maxI = 13000.0

    my_dpi=96
    w=21.3
    h=16

    # get list of files
    filesList=glob.glob("???_???.tiff")    

    #check if number of files equals variables
    if len(filesList) != NPSteps*NASteps:
        print("Total number of images expected:", AllImages)
        print("Images in the directory:", len(filesList))
        print("Mismatch! Aborting...")
        sys.exit()

    # get all images to array
    print("Loading data..", end='', flush=True)
    allImgArr = np.array([np.array(Image.open(fname)) for fname in filesList])
    print('.done')

    n_z, n_y, n_x = allImgArr.shape   # get dimensions of the array
    print("Array shape: x=", n_x, " y=", n_y, "images:", n_z)
    print("Data size in memory: " + str(round(allImgArr.nbytes / 1024 / 1024,2)) + " Mb")
    #print("The array shape: " + str(xmax - xmin) + "x" + str(ymax - ymin))


    #xarray=np.arange(-(n_z-1)/2, (n_z-1)/2+1, 1)*AStepRad   
    #xarray = np.loadtxt(fname)  
    if os.path.isfile(xarrayFName):
      #  xarr = np.zeros(lines, np.float16)
        #global xarray
        xarray = np.loadtxt(xarrayFName)
        #print(xarray)
    else:
        print('no APosCalibrated.dat file found!')   

    print("\nFitting parabolas..", end='', flush=True)

    yarr=allImgArr[:, dy, dx]
    xarr=xarray
    # remove saturated values
    for j in reversed(range(n_z)):
        if yarr[j] > maxI:
            yarr = np.delete(yarr, j)
            xarr = np.delete(xarr, j)

    a, b, c=np.polyfit(xarr, yarr, 2)

    thedeg = -57.3*b/(2*a)
    er = c/a-b*b/(4*a*a)

    print("done")
    #print("a=", a)
    #print("b=", b)
    #print("c=", c)

    plt.figure(figsize=(w,h), dpi=my_dpi)
    plt.margins(0)
    plt.scatter(xarr, yarr, label='data')
    xx = np.linspace(xarr.min(), xarr.max(), 31)
    yy = a*xx**2 + b*xx + c
    plt.plot(xx, yy, label='fit', color='red')
    #plt.title("P=" + str(PMidPosDeg) + "  X=" + str(dx) + "  Y=" + str(dy) + " max=" + str(format(yarr.max(), '.0f'))) # +
    #    " sigm/max="+str(format(sigm, '.3f')) + " rad="+str(format(minparr[dz], '.3f')))
    #fname=str(format(dz, '02d')) + "_" + str(format(dx, '04d')) + "_" + str(format(dy, '04d')) + ".png"
    plt.tight_layout(pad=5)
    plt.show
    #plt.savefig("test.png")
    #plt.close()
