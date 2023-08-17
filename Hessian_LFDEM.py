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
import itertools
import re
from scipy.spatial.distance import pdist
from os import listdir
from os.path import isfile, join
import time
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from numpy import linalg as LA
from scipy import sparse

def hessianDataExtractor(intFile,parFile,dataFile,numParticles,snapShotRange=False):
    """
    This function extracts the needed data for the Hessian which is constructed in a subsequent function, hessianGenerator.
    In order to calculate the frictional hessian we need the IDs of each particle in contact, the two spring
    constants associated with the contact, their interparticle separation, the normal vector, and the particle radii's.
    Below outlines where that information comes from in the output of LF_DEM
    
    1.  Particle radii are retrieved from the parFile.
    
    2.  We'll first get the spring constants for each strain step from the dataFile.
    
    3.  The intFile contains the particle IDs of the particles in contact, their contact state (sliding or not), 
    as well as normal vector components and the center to center distance.

    Inputs:
    intFile:  This is the path to the int_ file.
    parFile:  This is the path to the par_ file.
    dataFile:  This is the path to the data_ file.
    numParticles:  This is the number of particles in the system, usually it is numParticles=2000
    snapShotRange:  This is an optional argument that you can use to extract the data needed for the hessian for only
    specific simulation snapshots as the par_ and int_ files will usually contain >200 snapshots and that can take awhile.
    If snapShotRange=False then this program will output the data for every snapshot.  If snapShotRange=[0,5] that will
    process the first 5 snapshots and if snapShotRange=[50,-1] then this function will begin at the 51st snapshot and
    keep processing until it reaches the last snapshot.

    Outputs:
    radii:  This is a list of particle radii's.
    springConstantsNew:  This is an array that has three columns and a number of rows that corresponds to the number of
    snapshots.  The first column is the strain that the snapshot corresponds to, the second column is the k_n spring
    constant of that snapshot, and the final column is the k_t spring constant of that snapshot
    contactInfo:  This is a list whose size is the number of snapshots processed.  Each item in the list is a numpy array
    that has six columns and a number of rows that corresponds to the number of frictional particle interactions in that snapshot.
    Where the first column is the first particle index, the second is the second particle index,
    the third column tells us the contact type, the fourth column is the first component of the normal vector,
    the fifth column is the second component of the normal vector, and finally the last column is the dimensionless gap
    """
    
    #1.  Let's get the particle radii and the number of snapshots in the parFile.
    positionData = np.loadtxt(parFile,usecols=[1,2,4])
    numSnapshots = int(np.shape(positionData)[0]/numParticles)
    radii = positionData[:numParticles,0]

    #We don't need the positionData so let's delete it from RAM.  I'm pretty sure most compilers do this automatically
    #but it doesn't hurt to do it here.
    del positionData

    #If you don't want to process every single snapshot in the int_, par_, and data_ files you encode that in snapShotRange
    #as outlined above in the header of this function.
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
    
    #We're only extracting the strain step, k_n, and k_t from the data files so we'll make a holder for those values
    springConstants = np.zeros((len(fileLines),3))

    #This loops over each line and in the data file and extracts the strain step, k_n, and k_t, placing them into the holder
    #we created above.
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
    strainInfo = np.zeros(numSnapshots)
    linesWhereDataStarts=np.array([])
    for lines in fileLines:
        if (np.shape(np.fromstring(lines,sep=' '))[0]==0) & ("#" in str(lines)):
            strainInfo[otherCounter] = np.fromstring(lines.replace("#",""),sep=' ')[0]
            linesWhereDataStarts = np.append(linesWhereDataStarts, counter)
            otherCounter = otherCounter +1
            
        counter=counter+1
        
        
    #Now lets make a python list to store all the different contact information for each snapshot
    contactInfo = [0] * (upperSnapShotRange-lowerSnapShotRange)
    strainsFromSnapShot = [0] * (upperSnapShotRange-lowerSnapShotRange)

    #Now we'll loop through each snapshot in the int_ file and extract the particle indices, the contact type (sliding or not),
    #the two components of the normal vector, and finally the dimensionless gap between the particles.  All of this will be stored in
    #the array "currentContacts" which will be inputted into "contactInfo" at the end of each iteration of the loop.
    #The format of "currentContacts" first column is the first particle index, the second is the second particle index,
    #the third column tells us the contact type, the fourth column is the first component of the normal vector,
    #the fifth column is the second component of the normal vector, and finally the last column is the dimensionless gap
    counter = 0
    for i in range(lowerSnapShotRange,upperSnapShotRange):
        if i==numSnapshots-1:
            #This is the edge case when i is equal to the last snapshot.
            strainsFromSnapShot[counter] = strainInfo[i]
            currentContacts = np.genfromtxt(itertools.islice(fileLines, int(linesWhereDataStarts[i]), int(numLines)),usecols=(0,1,2,3,5,6))
            #If there is a 0 or 1 in the third column then that means the particles are not in contact and we can throw that row out.
            currentContacts = currentContacts[np.where(currentContacts[:, 2] > 1), :][0]
            if len(currentContacts)==0:
                contactInfo[counter] = [0]
            else:    
                contactInfo[counter] = currentContacts       
        else:
            strainsFromSnapShot[counter] = strainInfo[i]
            currentContacts = np.genfromtxt(itertools.islice(fileLines, int(linesWhereDataStarts[i]), int(linesWhereDataStarts[i + 1])),usecols=(0,1,2,3,5,6))
            currentContacts = currentContacts[np.where(currentContacts[:, 2] > 1), :][0]
            if len(currentContacts)==0:
                contactInfo[counter] = [0]
            else:    
                contactInfo[counter] = currentContacts
        counter=counter+1


    #Before  returning the data we have too many spring constants as the data_ files contain ~10X more data than the int_ and par_ files.
    springConstantsNew = np.zeros(((upperSnapShotRange-lowerSnapShotRange),3))
    counter=0
    for s in springConstants:
        if s[0] in strainsFromSnapShot:
            springConstantsNew[counter,:] = s
            counter=counter+1
            
            
    

    return radii, springConstantsNew, contactInfo

def hessianGenerator(radii,springConstants,contactInfo,outputDir,nameOfFile,snapShotRange=False,partialSave=False,cylinderHeight=1,particleDensity=1):
    
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
        5.  nameOfFile:  This is the name of the file that will be outputted into outputDir
        6 snapShotRange:  This is an optional argument that you can use to extract the data needed for the hessian for only
        specific simulation snapshots as the par_ and int_ files will usually contain >200 snapshots and that can take awhile.
        If snapShotRange=False then this program will output the data for every snapshot.  If snapShotRange=[0,5] that will
        process the first 5 snapshots and if snapShotRange=[50,-1] then this function will begin at the 51st snapshot and
        keep processing until it reaches the last snapshot.
        7 partialSave:  This outputs hessianFiles more so that the RAM doesn't get bogged down by holding all the data at once.
    """
    snapShotArray = np.linspace(snapShotRange[0],snapShotRange[1],snapShotRange[1]-snapShotRange[0]+1)
    lastSaveSnapShot = snapShotRange[0]
    #Each Hessian file takes up a lot of spac and
    if os.path.exists(os.path.join(outputDir,"hessianFiles"))==False:
        os.mkdir(os.path.join(outputDir,"hessianFiles"))




    #Most entries in the Hessian will be zero.  This is great for us because it means
    #it will be faster to just store the value of the entry and the three (i,j,k) values
    #that correspond to the indices of the entries.  Indices will start at 0.

    hessianRowHolder = np.array([])
    hessianColHolder = np.array([])
    hessian3rdDimHolder = np.array([])
    hessianDataHolder = np.array([])
    
    globalTimer = time.time()
    counter=0
    springConstantCounter=0
    for currentSnapShot in contactInfo:
        print("Starting new snapshot, currently this calculation has been running for:" +str((time.time()-globalTimer)/60) +" minutes")
        #Let's get the spring constants for this snapshot
        kn = springConstants[springConstantCounter,1]
        k_t = springConstants[springConstantCounter,2]
        snapShotIndex = snapShotArray[counter]
        
        
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
                #and just relabeling things.  The other thing is that we have to 
                #reconfigure everything so that it is in a coo_array format


                # This is our litte square in local coordinates (where nonzero)
                subsquare = np.zeros((3, 3))
                subsquare[0, 0] = -kn
                subsquare[1, 1] = fn / rval - kt
                subsquare[1, 2] = kt * Aj
                # note asymmetric cross-term
                subsquare[2, 1] = -kt * Ai
                subsquare[2, 2] = kt * Ai * Aj

                #Mike:  Since we're no longer sticking it in one huge matrix I'm instead encoding these values into
                #the sparse array format
                # Stick this into the appropriate places after rotating it away from the (n,t) frame
                #Hij = np.zeros((3, 3))
                #Hij[0, 0] = subsquare[0, 0] * nx0**2 + subsquare[1, 1] * tx0**2
                #Hij[0,1] = subsquare[0, 0] * nx0 * ny0 + subsquare[1, 1] * tx0 * ty0
                #Hij[1,0] = subsquare[0, 0] * ny0 * nx0 + subsquare[1, 1] * ty0 * tx0
                #Hij[1, 1] = subsquare[0, 0] * ny0**2 + subsquare[1, 1] * ty0**2
                #Hij[0, 2] = subsquare[1, 2] * tx0
                #Hij[1, 2] = subsquare[1, 2] * ty0
                #Hij[2, 0] = subsquare[2, 1] * tx0
                #Hij[2, 1] = subsquare[2, 1] * ty0
                #Hij[2, 2] = subsquare[2, 2]
                
                #Mike:  Here is the sparse matrix format of what is above.
                hessianRowHolder = np.append([3*i,3*i,3*i+1,3*i+1,3*i,3*i+1,3*i+2,3*i+2,3*i+2],hessianRowHolder)
                hessianColHolder = np.append([3*j,3*j+1,3*j,3*j+1,3*j+2,3*j+2,3*j,3*j+1,3*j+2],hessianColHolder)
                currentMatCValues = [subsquare[0, 0] * nx0**2 + subsquare[1, 1] * tx0**2,subsquare[0, 0] * nx0 * ny0 + subsquare[1, 1] * tx0 * ty0,subsquare[0, 0] * ny0 * nx0 + subsquare[1, 1] * ty0 * tx0,subsquare[0, 0] * ny0**2 + subsquare[1, 1] * ty0**2,subsquare[1, 2] * tx0,subsquare[1, 2] * ty0,subsquare[2, 1] * tx0,subsquare[2, 1] * ty0,subsquare[2, 2]]
                hessianDataHolder = np.append(currentMatCValues/ (mi * mj)**0.5,hessianDataHolder)
                
    
                # And put it into the Hessian, with correct elasticity prefactor
                # once for contact ij
                #hessianHolder[3 * i:(3 * i + 3),3 * j:(3 * j + 3),counter] = Hij / (mi * mj)**0.5
    
                # see notes for the flip one corresponding to contact ji
                # both n and t flip signs. Put in here explicitly. Essentially, angle cross-terms flip sign
                # Yes, this is not fully efficient, but it's clearer. Diagonalisation is rate-limiting step, not this.
                # careful with the A's
                subsquare[1, 2] = subsquare[1, 2] * Ai / Aj
                subsquare[2, 1] = subsquare[2, 1] * Aj / Ai
                #Hji = np.zeros((3, 3))
                #Hji[0,0] = subsquare[0, 0] * (-nx0)**2 + subsquare[1, 1] * (-tx0)**2
                #Hji[0, 1] = subsquare[0, 0] * (-nx0) * (-ny0) + subsquare[1, 1] * (-tx0) * (-ty0)
                #Hji[1, 0] = subsquare[0, 0] * (-ny0) * (-nx0) + subsquare[1, 1] * (-ty0) * (-tx0)
                #Hji[1,1] = subsquare[0, 0] * (-ny0)**2 + subsquare[1, 1] * (-ty0)**2
                #Hji[0, 2] = subsquare[1, 2] * (-tx0)
                #Hji[1, 2] = subsquare[1, 2] * (-ty0)
                #Hji[2, 0] = subsquare[2, 1] * (-tx0)
                #Hji[2, 1] = subsquare[2, 1] * (-ty0)
                #Hji[2, 2] = subsquare[2, 2]

                # Mike:  Here is the sparse matrix format of what is above.
                hessianRowHolder = np.append([3*j,3*j,3*j+1,3*j+1,3*j,3*j+1,3*j+2,3*j+2,3*j+2],hessianRowHolder)
                hessianColHolder = np.append([3*i,3*i+1,3*i,3*i+1,3*i+2,3*i+2,3*i,3*i+1,3*i+2],hessianColHolder)
                currentMatCValues = [subsquare[0, 0] * (-nx0)**2 + subsquare[1, 1] * (-tx0)**2,subsquare[0, 0] * (-nx0) * (-ny0) + subsquare[1, 1] * (-tx0) * (-ty0),subsquare[0, 0] * (-ny0) * (-nx0) + subsquare[1, 1] * (-ty0) * (-tx0),subsquare[0, 0] * (-ny0)**2 + subsquare[1, 1] * (-ty0)**2,subsquare[1, 2] * (-tx0),subsquare[1, 2] * (-ty0),subsquare[2, 1] * (-tx0),subsquare[2, 1] * (-ty0),subsquare[2, 2]]
                hessianDataHolder = np.append(currentMatCValues /(mi * mj)**0.5,hessianDataHolder)
                # And put it into the Hessian
                # now for contact ji
                #hessianHolder[3 * j:(3 * j + 3),3 * i:(3 * i + 3),counter] = Hji / (mi * mj)**0.5
    
                # Careful, the diagonal bits are not just minus because of the rotations
                diagsquare = np.zeros((3, 3))
                diagsquare[0, 0] = kn
                diagsquare[1, 1] = -fn / rval + kt
                diagsquare[1, 2] = kt * Ai
                diagsquare[2, 1] = kt * Ai
                diagsquare[2, 2] = kt * Ai**2
    
                # Stick this into the appropriate places:
                #Hijdiag = np.zeros((3, 3))
                #Hijdiag[0, 0] = diagsquare[0, 0] * nx0**2 + diagsquare[1, 1] * tx0**2
                #Hijdiag[0, 1] = diagsquare[0, 0] * nx0 * ny0 + diagsquare[1, 1] * tx0 * ty0
                #Hijdiag[1, 0] = diagsquare[0, 0] * ny0 * nx0 + diagsquare[1, 1] * ty0 * tx0
                #Hijdiag[1, 1] = diagsquare[0, 0] * ny0**2 + diagsquare[1, 1] * ty0**2
                #Hijdiag[0, 2] = diagsquare[1, 2] * tx0
                #Hijdiag[1, 2] = diagsquare[1, 2] * ty0
                #Hijdiag[2, 0] = diagsquare[2, 1] * tx0
                #Hijdiag[2, 1] = diagsquare[2, 1] * ty0
                #Hijdiag[2, 2] = diagsquare[2, 2]
    
                # And then *add* it to the diagnual
                hessianRowHolder = np.append([3*i,3*i,3*i+1,3*i+1,3*i,3*i+1,3*i+2,3*i+2,3*i+2],hessianRowHolder)
                hessianColHolder = np.append([3*i,3*i+1,3*i,3*i+1,3*i+2,3*i+2,3*i,3*i+1,3*i+2],hessianColHolder)
                currentMatCValues = [diagsquare[0, 0] * nx0**2 + diagsquare[1, 1] * tx0**2,diagsquare[0, 0] * nx0 * ny0 + diagsquare[1, 1] * tx0 * ty0,diagsquare[0, 0] * ny0 * nx0 + diagsquare[1, 1] * ty0 * tx0,diagsquare[0, 0] * ny0**2 + diagsquare[1, 1] * ty0**2,diagsquare[1, 2] * tx0,diagsquare[1, 2] * ty0,diagsquare[2, 1] * tx0,diagsquare[2, 1] * ty0,diagsquare[2, 2]]
                hessianDataHolder = np.append(currentMatCValues / mi,hessianDataHolder)

                #hessianHolder[3 * i:(3 * i + 3), 3 * i:(3 * i + 3),counter] += Hijdiag / mi
    
                #And once more for the jj contribution, which is the same whizz with the flipped sign of n and t
                # and adjusted A's
                diagsquare = np.zeros((3, 3))
                diagsquare[0, 0] = kn
                diagsquare[1, 1] = -fn / rval + kt
                diagsquare[1, 2] = kt * Aj
                diagsquare[2, 1] = kt * Aj
                diagsquare[2, 2] = kt * Aj**2
    
                #Hjidiag = np.zeros((3, 3))
                #Hjidiag[0, 0] = diagsquare[0, 0] * (-nx0)**2 + diagsquare[1, 1] * (-tx0)**2
                #jidiag[0, 1] = diagsquare[0, 0] * (-nx0) * (-ny0) + diagsquare[1, 1] * (-tx0) * (-ty0)
                #Hjidiag[1, 0] = diagsquare[0, 0] * (-ny0) * (-nx0) + diagsquare[1, 1] * (-ty0) * (-tx0)
                #Hjidiag[1, 1] = diagsquare[0, 0] * (-ny0)**2 + diagsquare[1, 1] * (-ty0)**2
                #Hjidiag[0, 2] = diagsquare[1, 2] * (-tx0)
                #Hjidiag[1, 2] = diagsquare[1, 2] * (-ty0)
                #Hjidiag[2, 0] = diagsquare[2, 1] * (-tx0)
                #Hjidiag[2, 1] = diagsquare[2, 1] * (-ty0)
                #Hjidiag[2, 2] = diagsquare[2, 2]
    
                # And then *add* it to the diagnual
                #hessianHolder[3 * j:(3 * j + 3), 3 * j:(3 * j + 3),counter] += Hjidiag / mj
                hessianRowHolder = np.append([3*j,3*j,3*j+1,3*j+1,3*j,3*j+1,3*j+2,3*j+2,3*j+2],hessianRowHolder)
                hessianColHolder = np.append([3*j,3*j+1,3*j,3*j+1,3*j+2,3*j+2,3*j,3*j+1,3*j+2],hessianColHolder)
                currentMatCValues = [diagsquare[0, 0] * (-nx0)**2 + diagsquare[1, 1] * (-tx0)**2,diagsquare[0, 0] * (-nx0) * (-ny0) + diagsquare[1, 1] * (-tx0) * (-ty0),diagsquare[0, 0] * (-ny0) * (-nx0) + diagsquare[1, 1] * (-ty0) * (-tx0),diagsquare[0, 0] * (-ny0)**2 + diagsquare[1, 1] * (-ty0)**2,diagsquare[1, 2] * (-tx0),diagsquare[1, 2] * (-ty0),diagsquare[2, 1] * (-tx0),diagsquare[2, 1] * (-ty0),diagsquare[2, 2]]
                hessianDataHolder = np.append(currentMatCValues / mj,hessianDataHolder)

                hessian3rdDimHolder = np.append(counter*np.ones(36),hessian3rdDimHolder)

        #This portion of the code checks to see what the counter is and if it's a multiple of 10 it saves
        if partialSave!=False:
            if counter%100==0 and counter!=0:

                nameOfFileNew=nameOfFile.replace("hessian_","hessian_"+str(lastSaveSnapShot)+"_"+str(snapShotIndex)+"_")
                np.save(nameOfFileNew,np.vstack((hessianRowHolder, hessianColHolder, hessian3rdDimHolder, hessianDataHolder)))
                hessianRowHolder = np.array([])
                hessianColHolder = np.array([])
                hessian3rdDimHolder = np.array([])
                hessianDataHolder = np.array([])
                lastSaveSnapShot = snapShotIndex

        counter=counter+1
        springConstantCounter = springConstantCounter+1

    nameOfFileNew = nameOfFile.replace("hessian_", "hessian_" + str(lastSaveSnapShot) + "_" + str(snapShotIndex) + "_")
    np.save(nameOfFileNew,np.vstack((hessianRowHolder, hessianColHolder,hessian3rdDimHolder,hessianDataHolder)))

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

        nameOfHessianFile = currentParFile.replace("par_","hessian_").replace(".dat","")
        hessianGenerator(radii,springConstants,currentContacts,topDir,nameOfHessianFile)


def generateSingleHessianFiles(topDir,stressValue,numParticles,snapShotRange=False,partialSave=False):

    
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
    
    dataFileIndex = np.where(dataStressHolder==stressValue)[0][0]
    intFileIndex = np.where(intStressHolder==stressValue)[0][0]
    parFileIndex = np.where(parStressHolder==stressValue)[0][0]


    intFile = intFileHolder[intFileIndex]
    dataFile = dataFileHolder[dataFileIndex]
    parFile = parFileHolder[parFileIndex]
    nameOfHessianFile = parFile.replace("par_","hessian_").replace(".dat","")

    radii, springConstants, currentContacts = hessianDataExtractor(intFile,parFile,dataFile,numParticles,snapShotRange)

    firstSnapShot = snapShotRange[0]
    lastSnapShot = snapShotRange[0]+len(currentContacts)
    hessianGenerator(radii,springConstants,currentContacts,topDir,nameOfHessianFile,[firstSnapShot,lastSnapShot],partialSave)



def eigenValueVectorExtraction(topDir,numParticles,numSnapShotsToProcess=-1):
    
    #Find all the files in topDir that start with "hessian_"
    hessianFileHolder = []
    for file in os.listdir(topDir):
        if file.startswith("hessian_"):
            hessianFileHolder.append(file)
    #Add topDir to the beginning of each hessian file found in topDir
    hessianFileHolderNoTopDir = hessianFileHolder
    hessianFileHolder = [os.path.join(topDir,file) for file in hessianFileHolder]
    
    
    #Loop over every hessian file found
    hessianCounter=0
    for currentHessianFile in hessianFileHolder:
        print("currently running" + currentHessianFile)
        
        #Load in the current Hessian and transpose it because otherwise I get
        #confused :(
        currentHessian = np.load(currentHessianFile)
        currentHessian = np.transpose(currentHessian)
        eigenValueName = hessianFileHolderNoTopDir[hessianCounter].replace("hessian_","eigenValues_").replace(".npy","").replace(topDir,"")
        eigenVectorName = hessianFileHolderNoTopDir[hessianCounter].replace("hessian_","eigenVector_").replace(".npy","").replace(topDir,"")
        
        #The format of each hessian file is the the first two rows 
        
    
        firstSnapShot = int(min(currentHessian[:,2]))
        if numSnapShotsToProcess==-1:
            lastSnapShot = int(max(currentHessian[:,2]))
        else:
            lastSnapShot = firstSnapShot+numSnapShotsToProcess
        
        #These files are so large we need to save them every iteration of the upcoming loop.
        #This way we aren't carrying around huge amounts of stuff in the RAM, the tradeoff is that we are going to be
        #saving files very often and lots of them so we should make a directory to hold them
        eigenDirName = currentHessianFile.replace("hessian_","eigenDirectory_").replace(".npy","")
        os.mkdir(os.path.join(eigenDirName))
        
        eigenValueHolder = np.zeros((lastSnapShot+1-firstSnapShot,3*numParticles))
        counter=0        
        for i in range(firstSnapShot,lastSnapShot+1):
            print("counter = " + str(counter))
            
            indicesOfInterest = np.where(currentHessian[:,2]==i)[0]
            hessianOfInterest = currentHessian[indicesOfInterest,:]
            hessianOfInterest = np.delete(hessianOfInterest, 2, 1)
            sparseMatrix = csr_matrix((hessianOfInterest[:,2], (hessianOfInterest[:,0], hessianOfInterest[:,1])), shape = (3*numParticles, 3*numParticles)).toarray()
            sparseMatrix = (0.5)*(sparseMatrix+np.transpose(sparseMatrix))
            
            eigenValueHolder, eigenVectorHolder = LA.eig(sparseMatrix)
            sparseEigenValues = sparse.csr_matrix(eigenValueHolder)
            sparseEigenVectors = sparse.csr_matrix(eigenVectorHolder)
            
            sparse.save_npz(os.path.join(eigenDirName,eigenValueName+str(counter)), sparseEigenValues)
            sparse.save_npz(os.path.join(eigenDirName,eigenVectorName+str(counter)), sparseEigenVectors)

            counter=counter+1
    
        hessianCounter=hessianCounter+1
        #eigenValueHolder = sparse.csr_matrix(eigenValueHolder)
        
        #sparse.save_npz(eigenValueName, eigenValueHolder)


