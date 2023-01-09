# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 09:45:34 2022
@author: Mike van der Naald

This code is based on Silke Henke's code called "Hessian.py" found in the following Github repository:
    
    
    
That code takes experimental data and also frictional packing simulations as inputs
and generates their frictional Hessian, more information can be found in the following publication:
    
    

This code takes in LF_DEM simulation data as inputs and generates their frictional Hessian.


"""
import numpy as np
import sys
import os
from matplotlib import pyplot
import Configuration_LFDEM as CF #Mike:  I modified the Configuration class so we don't have to load in using text files.
import Pebbles as PB
import Hessian as HS
import Analysis as AN
import itertools
import re
from scipy.spatial.distance import pdist
from os import listdir
from os.path import isfile, join

def hessianDataExtractor(intFile,parFile,dataFile,numParticles,snapShotRange=False):
    """
    This function extracts the needed data for the Hessian which is constructed in a subsequent function.
    In order to calculate the frictional hessian we need the IDs of each particle in contact, the two spring
    constants associated with the contact, their interparticle separation, the normal vector, the tangent vector,
    and the particle radii's.
    
    1.  Particle radii are retrieved from the parFile.
    
    2.  We'll first get the spring constants for each strain step from the dataFile.
    
    3.  The intFile contains the particle IDs of the particles in contact, their contact state (sliding or not), 
    as well as normal vector components and the center to center distance.
    """
    
    #1.  Let's get the particle radii and the number of snapshots in the parFile.
    positionData = np.loadtxt(parFile,usecols=[1,2,4])
    numSnapshots = int(np.shape(positionData)[0]/numParticles)
    radii = positionData[:numParticles,0]
    
    del positionData
    
    if snapShotRange == False:
        lowerSnapShotRange = 0
        upperSnapShotRange = numSnapshots
    else:
        lowerSnapShotRange = snapShotRange[0]
        upperSnapShotRange = snapShotRange[1]
        if upperSnapShotRange == -1:
            upperSnapShotRange=numSnapshots
    
    #2.  Let's get the spring constants from the data file
    with open(dataFile) as f1:
        fileLines = f1.readlines()[45:]
    
    #We're only extracting the strain step, k_n, and k_t from the data files
    springConstants = np.zeros((len(fileLines),3))
    counter=0
    for lines in fileLines:
        currentLine = np.fromstring(str(lines.replace("n","0")),dtype=np.double, sep=' ')
        springConstants[counter,:] = np.array([currentLine[1],currentLine[31],currentLine[32]])
        counter=counter+1
        
    #3.  Let's extract which particles are in contact, their separations, and from the intFile.
    #Ignore the header lines (first 25 lines).
    with open(intFile) as f1:
        fileLines = f1.readlines()[24:]

    numLines = np.shape(fileLines)[0]

    #We'll first find every line in intFile that starts with "#" as that is a line where a new snapshop starts.
    counter=0
    otherCounter=0
    strainInfo = np.zeros(upperSnapShotRange-lowerSnapShotRange)
    linesWhereDataStarts=np.array([])
    for lines in fileLines:
        if (np.shape(np.fromstring(lines,sep=' '))[0]==0) & ("#" in str(lines)):
            strainInfo[otherCounter] = np.fromstring(lines.replace("#",""),sep=' ')[0]
            linesWhereDataStarts = np.append(linesWhereDataStarts, counter)
            otherCounter = otherCounter +1
            
        counter=counter+1
        
        
    #Now lets make a python list to store all the different contacts for each snapshot
    contactInfo = [0] * (upperSnapShotRange-lowerSnapShotRange)

    #Now we'll loop through each snapshot and store only the first three columns.  This should hopefully make this less expensive.
    #The first column is the first particle index, the second is the second particle index and the final column tells us the contact type.
    #We will also be ignoring any interaction where the contact type is 0 as that is a hydrodynamic interaction.
    counter = 0
    for i in range(lowerSnapShotRange,upperSnapShotRange):
        if i==numSnapshots-1:
            #If there is a 0 in the third column then that means the particles are not in contact and we can throw that row our.
            currentContacts = np.genfromtxt(itertools.islice(fileLines, int(linesWhereDataStarts[i]), int(numLines)),usecols=(0,1,2,3,5,6))
            currentContacts = currentContacts[np.where(currentContacts[:, 2] > 1), :][0]
            if len(currentContacts)==0:
                contactInfo[counter] = [0]
            else:    
                contactInfo[counter] = currentContacts       
        else:
            currentContacts = np.genfromtxt(itertools.islice(fileLines, int(linesWhereDataStarts[i]), int(linesWhereDataStarts[i + 1])),usecols=(0,1,2,3,5,6))
            currentContacts = currentContacts[np.where(currentContacts[:, 2] > 1), :][0]
            if len(currentContacts)==0:
                contactInfo[counter] = [0]
            else:    
                contactInfo[counter] = currentContacts
        counter=counter+1


    #Before reporting returning the data we have too many spring constants as the data_ files contain ~10X more data than the int_ files.
    springConstantsNew = np.zeros((numSnapshots,3))
    counter=0
    for s in springConstants:
        if s[0] in strainInfo:
            springConstantsNew[counter,:] = s
            counter=counter+1
            
            
    
    
    return radii, springConstantsNew, contactInfo

def hessianGenerator(radii,springConstants,contactInfo,outputDir,stressValue,cylinderHeight=1,particleDensity=1,snapShotsPerBatch=50):
    
    """
    This function generates a 2D frictional Hessian using the formulation and some original code from the following publication:
        
        
        
    The three input variables are:
        1.  The radii, this is a numpy array that has an entry for each particle and denotes what their radii is.
        2.  springConstants is a numpy array that has a row for each simulation snap shot.  The first column is the 
        strain step of the snap shot, the second column is the normal spring constant coeff, and the third column is the
        tangential spring constant coeff.
        3.  contactInfo is a list with each entry corresponding to a numpy array for each simulation snapshot.  Each numpy array
        has a row for each frictional contact, the first column is the ID of the particle i, the second column is the ID of particle j.
        The third and fourth columns are the x and z components of the normal vector (y is zero in 2D).  Finally the fifth entry is the 
        "dimensionless gap which we'll convert into the center to center distance.
        4.  outputDir is the directory to output the Hessian to.
        5.  stressValue is the value of the applied stress for this set of systems.
    """
    
    numParticles = len(radii)
    
    #The hessian holder needs to be split into multiple arrays and saved to disk multiple times.  This is because an array of size
    #[3*numParticles, 3*numParticles, numSnapshots] will have 86400000000000 entries for numParticles=2000 and numSnapshots=400 which is too large
    #to have in RAM.  We will write it out every 100 snapshots.
    
    
    hessianHolder = np.zeros((3 * numParticles, 3 * numParticles,snapShotsPerBatch))
    
    counter=0
    hessianCounter=0
    springConstantCounter=0
    for currentSnapShot in contactInfo:
        #Let's get the spring constants for this snapshot
        kn = springConstants[springConstantCounter,1]
        k_t = springConstants[springConstantCounter,2] 
        
        
        #Now let's loop through every contact and construct that portion of the Hessian.
        for contact in currentSnapShot:
            if np.array_equal(currentSnapShot,np.array([0]))!=True:
                
                i = int(contact[0])
                j = int(contact[1])
                slidingOrNot = int(contact[2])
                nx0 = contact[3]
                ny0 = contact[4]
                dimGap = contact[5]
                
                tx0 = -ny0
                ty0 = nx0
                    
                R_i = radii[i]
                R_j = radii[j]
                rval = (1/2)*(dimGap+2)*(R_i+R_j)
                
                Ai = 1.0 / 2**0.5
                Aj = 1.0 / 2**0.5
                
                mi = particleDensity * cylinderHeight * np.pi * R_i**2
                mj = particleDensity * cylinderHeight * np.pi * R_j**2
                
                fn = kn * (R_i + R_j - dimGap)
                
                if slidingOrNot==2:
                    kt = k_t
                else:
                    kt = 0
                
                
                
                #Mike:  From here on out we're trusting Silke's original construction
                #and just relabeling things.
    
        
                # This is our litte square in local coordinates (where nonzero)
                subsquare = np.zeros((3, 3))
                subsquare[0, 0] = -kn
                subsquare[1, 1] = fn / rval - kt
                subsquare[1, 2] = kt * Aj
                # note asymmetric cross-term
                subsquare[2, 1] = -kt * Ai
                subsquare[2, 2] = kt * Ai * Aj
    
                # Stick this into the appropriate places after rotating it away from the (n,t) frame
                Hij = np.zeros((3, 3))
                Hij[0, 0] = subsquare[0, 0] * nx0**2 + subsquare[1, 1] * tx0**2
                Hij[0,
                    1] = subsquare[0, 0] * nx0 * ny0 + subsquare[1, 1] * tx0 * ty0
                Hij[1,
                    0] = subsquare[0, 0] * ny0 * nx0 + subsquare[1, 1] * ty0 * tx0
                Hij[1, 1] = subsquare[0, 0] * ny0**2 + subsquare[1, 1] * ty0**2
                Hij[0, 2] = subsquare[1, 2] * tx0
                Hij[1, 2] = subsquare[1, 2] * ty0
                Hij[2, 0] = subsquare[2, 1] * tx0
                Hij[2, 1] = subsquare[2, 1] * ty0
                Hij[2, 2] = subsquare[2, 2]
    
                # And put it into the Hessian, with correct elasticity prefactor
                # once for contact ij
                hessianHolder[3 * i:(3 * i + 3),3 * j:(3 * j + 3),counter] = Hij / (mi * mj)**0.5
    
                # see notes for the flip one corresponding to contact ji
                # both n and t flip signs. Put in here explicitly. Essentially, angle cross-terms flip sign
                # Yes, this is not fully efficient, but it's clearer. Diagonalisation is rate-limiting step, not this.
                # careful with the A's
                subsquare[1, 2] = subsquare[1, 2] * Ai / Aj
                subsquare[2, 1] = subsquare[2, 1] * Aj / Ai
                Hji = np.zeros((3, 3))
                Hji[0,
                    0] = subsquare[0, 0] * (-nx0)**2 + subsquare[1, 1] * (-tx0)**2
                Hji[0, 1] = subsquare[0, 0] * (-nx0) * (-ny0) + subsquare[1, 1] * (
                    -tx0) * (-ty0)
                Hji[1, 0] = subsquare[0, 0] * (-ny0) * (-nx0) + subsquare[1, 1] * (
                    -ty0) * (-tx0)
                Hji[1,
                    1] = subsquare[0, 0] * (-ny0)**2 + subsquare[1, 1] * (-ty0)**2
                Hji[0, 2] = subsquare[1, 2] * (-tx0)
                Hji[1, 2] = subsquare[1, 2] * (-ty0)
                Hji[2, 0] = subsquare[2, 1] * (-tx0)
                Hji[2, 1] = subsquare[2, 1] * (-ty0)
                Hji[2, 2] = subsquare[2, 2]
    
                # And put it into the Hessian
                # now for contact ji
                hessianHolder[3 * j:(3 * j + 3),3 * i:(3 * i + 3),counter] = Hji / (mi * mj)**0.5
    
                # Careful, the diagonal bits are not just minus because of the rotations
                diagsquare = np.zeros((3, 3))
                diagsquare[0, 0] = kn
                diagsquare[1, 1] = -fn / rval + kt
                diagsquare[1, 2] = kt * Ai
                diagsquare[2, 1] = kt * Ai
                diagsquare[2, 2] = kt * Ai**2
    
                # Stick this into the appropriate places:
                Hijdiag = np.zeros((3, 3))
                Hijdiag[0,
                        0] = diagsquare[0, 0] * nx0**2 + diagsquare[1, 1] * tx0**2
                Hijdiag[0, 1] = diagsquare[0, 0] * nx0 * ny0 + diagsquare[
                    1, 1] * tx0 * ty0
                Hijdiag[1, 0] = diagsquare[0, 0] * ny0 * nx0 + diagsquare[
                    1, 1] * ty0 * tx0
                Hijdiag[1,
                        1] = diagsquare[0, 0] * ny0**2 + diagsquare[1, 1] * ty0**2
                Hijdiag[0, 2] = diagsquare[1, 2] * tx0
                Hijdiag[1, 2] = diagsquare[1, 2] * ty0
                Hijdiag[2, 0] = diagsquare[2, 1] * tx0
                Hijdiag[2, 1] = diagsquare[2, 1] * ty0
                Hijdiag[2, 2] = diagsquare[2, 2]
    
                # And then *add* it to the diagnual
                hessianHolder[3 * i:(3 * i + 3), 3 * i:(3 * i + 3),counter] += Hijdiag / mi
    
                #And once more for the jj contribution, which is the same whizz with the flipped sign of n and t
                # and adjusted A's
                diagsquare = np.zeros((3, 3))
                diagsquare[0, 0] = kn
                diagsquare[1, 1] = -fn / rval + kt
                diagsquare[1, 2] = kt * Aj
                diagsquare[2, 1] = kt * Aj
                diagsquare[2, 2] = kt * Aj**2
    
                Hjidiag = np.zeros((3, 3))
                Hjidiag[0, 0] = diagsquare[0, 0] * (-nx0)**2 + diagsquare[1, 1] * (
                    -tx0)**2
                Hjidiag[0, 1] = diagsquare[0, 0] * (-nx0) * (
                    -ny0) + diagsquare[1, 1] * (-tx0) * (-ty0)
                Hjidiag[1, 0] = diagsquare[0, 0] * (-ny0) * (
                    -nx0) + diagsquare[1, 1] * (-ty0) * (-tx0)
                Hjidiag[1, 1] = diagsquare[0, 0] * (-ny0)**2 + diagsquare[1, 1] * (
                    -ty0)**2
                Hjidiag[0, 2] = diagsquare[1, 2] * (-tx0)
                Hjidiag[1, 2] = diagsquare[1, 2] * (-ty0)
                Hjidiag[2, 0] = diagsquare[2, 1] * (-tx0)
                Hjidiag[2, 1] = diagsquare[2, 1] * (-ty0)
                Hjidiag[2, 2] = diagsquare[2, 2]
    
                # And then *add* it to the diagnual
                hessianHolder[3 * j:(3 * j + 3), 3 * j:(3 * j + 3),counter] += Hjidiag / mj
        
        counter=counter+1
        springConstantCounter = springConstantCounter+1
        if counter == snapShotsPerBatch-1:
            #The hessian holder is now full and we need to write it to disk and 
            #start a new hessian holder for the rest of the snapshots
            np.save(os.path.join(outputDir,str(stressValue)+"_cl_2D_N"+str(numParticles)+"_"+str(hessianCounter)+".dat"),hessianHolder)
            hessianCounter=hessianCounter+1
                    
            hessianHolder = np.zeros((3 * numParticles, 3 * numParticles,100))
            counter=0
        
        
        
def generateHessianFiles(topDir,numParticles,snapShotRange=False):

    
    #hessianDataExtractor(intFile,parFile,dataFile,numParticles,snapShotRange=False)
    #First let's collect all the par, int, and data  files needed from topDir
    intFileHolder = []
    parFileHolder = []
    dataFileHolder = []
    for file in os.listdir(topDir):
        if file.startswith("int_"):
            intFileHolder.append(file)
        if file.startswith("par_"):
            parFileHolder.append(file)
        if file.startswith("data_"):
            dataFileHolder.append(file)
            
    intFileHolder = [os.path.join(topDir,file) for file in intFileHolder]
    parFileHolder = [os.path.join(topDir,file) for file in parFileHolder]
    dataFileHolder = [os.path.join(topDir,file) for file in dataFileHolder]


    if len(intFileHolder)!=len(parFileHolder):
        raise Exception("There are an unequal number of par_ and int_ files in the topDir given.")
    if len(intFileHolder)!=len(dataFileHolder):
        raise Exception("There are an unequal number of data_ and int_ files in the topDir given.")
    if len(parFileHolder)!=len(dataFileHolder):
        raise Exception("There are an unequal number of data_ and par_ files in the topDir given.")
    
    intStressHolder = np.zeros(len(intFileHolder))
    parStressHolder = np.zeros(len(parFileHolder))
    dataStressHolder = np.zeros(len(dataFileHolder))
    
    for i in range(0,len(intFileHolder)):     
        result = re.search('_stress(.*)cl', intFileHolder[i])
        intStressHolder[i] = float(result.group(1))
        result = re.search('_stress(.*)cl', parFileHolder[i])
        parStressHolder[i] = float(result.group(1))
        result = re.search('_stress(.*)cl', dataFileHolder[i])
        dataStressHolder[i] = float(result.group(1))
        
        
    #Now that we have all the int and par files we can begin iterating through them to generate the hessian files
    for i in range(0,len(intFileHolder)):
        currentIntFile = intFileHolder[i]
        currentStress = intStressHolder[i]
        currentDataFile = dataFileHolder[int(np.where(dataStressHolder==currentStress)[0][0])]
        currentParFile = parFileHolder[int(np.where(parStressHolder==currentStress)[0][0])]

  
        radii, springConstants, currentContacts = hessianDataExtractor(currentIntFile,currentParFile,currentDataFile,numParticles,snapShotRange)

        hessianGenerator(radii,springConstants,currentContacts,topDir,currentStress)
        

        
        
    
