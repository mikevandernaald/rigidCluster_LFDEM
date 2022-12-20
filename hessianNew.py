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



parFile = r"D:\data\rigidClusterData\dataFromCluster\mu_0.1\VF0.79\par_D2N2000VF0.79Bidi1.4_0.5Square_1_pardata_phi0.79_stress10cl.dat"
intFile = r"D:\data\rigidClusterData\dataFromCluster\mu_0.1\VF0.79\int_D2N2000VF0.79Bidi1.4_0.5Square_1_pardata_phi0.79_stress10cl.dat"
dataFile = r"D:\data\rigidClusterData\dataFromCluster\mu_0.1\VF0.79\data_D2N2000VF0.79Bidi1.4_0.5Square_1_pardata_phi0.79_stress10cl.dat"
numParticles = 2000
snapShotRange=[0,-1]


def hessianDataExtractor(intFile,parFile,dataFile,numParticles,snapShotRange=False):
    """
    This function extracts the needed data for the Hessian which is constructed in a subsequent function.
    In order to calculate the frictional hessian we need the IDs of each particle in contact, the two spring
    constants associated with the contact, their interparticle separation, the normal vector, the tangent vector,
    and the particle radii's.
    
    1.  Particle radii are retrieved from the parFile.
    
    2.  We'll first get the spring constants for each strain step from the dataFile.
    
    3.  We'll get the list of particle contacts, their IDs, interparticle separation, and .
    
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
    
    
    
    


dimGap  = contactInfo[6][4,6]
partI = int(contactInfo[6][4,0])
partJ = int(contactInfo[6][4,1])

posI = positionData[partI,:,6]
posJ = positionData[partJ,:,6]


Ri = radii[partI]
Rj = radii[partJ]

distance = np.sqrt(np.sum((posI-posJ)**2))

#s-2, s = 2r/(a1+a2)

s = dimGap+2

r = (s/2)*(Ri+Rj) #THIS IS THE TRUE DIM GAP




radii, springConstantsNew, currentContacts = hessianDataExtractor(intFile,parFile,dataFile,numParticles,snapShotRange)


def 2DHessianGenerator(radii,springConstants,contactInfo,particleDensity=1,cylinderHeight=1):
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
        dimensionless gap which we'll convert into the center to center distance.
    """
    numParticles = len(radii)
    numSnapshots = len(contactInfo)
    
    
    hessianHolder = np.zeros((3 * numParticles, 3 * numParticles,numSnapshots))
    
    
    counter=0
    for currentSnapShot in contactInfo:
        #Let's get the spring constants for this snapshot
        kn = springConstants[counter,1]
        k_t = springConstants[counter,2] 
        counter=counter+1
        
        
        #Now let's loop through every contact and construct that portion of the Hessian.
        for contact in currentSnapShot:
            
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
            particleOverlap = (1/2)*(dimGap+2)*(R_i+R_j)
            
            Ai = 1.0 / 2**0.5
            Aj = 1.0 / 2**0.5
            
            mi = particleDensity * cylinderHeight * np.pi * R_i**2
            mj = particleDensity * cylinderHeight * np.pi * R_j**2
            
            fn = kn * (R_i + R_j - particleOverlap)
            
            if slidingOrNot==2:
                kt = k_t
            else:
                kt = 0
            
            
            
            subsquare = np.zeros((3, 3))
            subsquare[0, 0] = -kn
            subsquare[1, 1] = fn / particleOverlap - kt
            subsquare[1, 2] = kt * Aj
            # note asymmetric cross-term
            subsquare[2, 1] = -kt * Ai
            subsquare[2, 2] = kt * Ai * Aj
            #collect our little forces
            favx += fn * nx0 + self.conf.ftan[k] * tx0
            favy += fn * ny0 + self.conf.ftan[k] * ty0
            frotav += self.conf.rad[i] * self.conf.ftan[k]

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
            hessianHolder[3 * i:(3 * i + 3),
                         3 * j:(3 * j + 3),counter] = Hij / (mi * mj)**0.5
                    
                
            
            
        
    
    
    for contacts in contactInformation:
        #Extract all the relevant information
        i = contacts[0]
        j = contacts[1]
        n_x = contacts[2]
        n_y = contacts[3]
        particleOverlap = contacts[4]
        
        #Extract radii so we can construct particle masses
        R_i = radii[i]
        R_j = radii[j]
        particleOverlap = (1/2)*(particleOverlap+2)*(R_i+R_j)
        
        
        mi = particleDensity * cylinderHeight * np.pi * R_i**2
        mj = particleDensity * cylinderHeight * np.pi * R_j**2
        
        Ai = 1.0 / 2**0.5
        Aj = 1.0 / 2**0.5
        
        
        tx0 = -n_y
        ty0 = n_x
        
        fn = K_n * (R_i+R_j -particleOverlap)
        
        
        dx = self.conf.x[j] - self.conf.x[i]
        
        dx -= self.conf.Lx * round(dx / self.conf.Lx)
        
        dy = self.conf.y[j] - self.conf.y[i]
        
        dy -= self.conf.Ly * round(dy / self.conf.Ly)
        rval = np.sqrt(dx**2 + dy**2)
        
        rval = np.sqrt(dx**2 + dy**2)

    
        # Stick this into the appropriate places after rotating it away from the (n,t) frame
        Hij = np.zeros((3, 3))
        Hij[0, 0] = subsquare[0, 0] * nx0**2 + subsquare[1, 1] * tx0**2
        Hij[0,1] = subsquare[0, 0] * nx0 * ny0 + subsquare[1, 1] * tx0 * ty0
        Hij[1,0] = subsquare[0, 0] * ny0 * nx0 + subsquare[1, 1] * ty0 * tx0
        Hij[1, 1] = subsquare[0, 0] * ny0**2 + subsquare[1, 1] * ty0**2
        Hij[0, 2] = subsquare[1, 2] * tx0
        Hij[1, 2] = subsquare[1, 2] * ty0
        Hij[2, 0] = subsquare[2, 1] * tx0
        Hij[2, 1] = subsquare[2, 1] * ty0
        Hij[2, 2] = subsquare[2, 2]

        # And put it into the Hessian, with correct elasticity prefactor
        # once for contact ij
        Hessian[3 * i:(3 * i + 3),3 * j:(3 * j + 3)] = Hij / (mi * mj)**0.5

        # see notes for the flip one corresponding to contact ji
        # both n and t flip signs. Put in here explicitly. Essentially, angle cross-terms flip sign
        # Yes, this is not fully efficient, but it's clearer. Diagonalisation is rate-limiting step, not this.
        # careful with the A's
        subsquare[1, 2] = subsquare[1, 2] * Ai / Aj
        subsquare[2, 1] = subsquare[2, 1] * Aj / Ai
        Hji = np.zeros((3, 3))
        Hji[0,0] = subsquare[0, 0] * (-nx0)**2 + subsquare[1, 1] * (-tx0)**2
        Hji[0, 1] = subsquare[0, 0] * (-nx0) * (-ny0) + subsquare[1, 1] * (-tx0) * (-ty0)
        Hji[1, 0] = subsquare[0, 0] * (-ny0) * (-nx0) + subsquare[1, 1] * (-ty0) * (-tx0)
        Hji[1,1] = subsquare[0, 0] * (-ny0)**2 + subsquare[1, 1] * (-ty0)**2
        Hji[0, 2] = subsquare[1, 2] * (-tx0)
        Hji[1, 2] = subsquare[1, 2] * (-ty0)
        Hji[2, 0] = subsquare[2, 1] * (-tx0)
        Hji[2, 1] = subsquare[2, 1] * (-ty0)
        Hji[2, 2] = subsquare[2, 2]

        # And put it into the Hessian
        # now for contact ji
        Hessian[3 * j:(3 * j + 3),3 * i:(3 * i + 3)] = Hji / (mi * mj)**0.5

        # Careful, the diagonal bits are not just minus because of the rotations
        diagsquare = np.zeros((3, 3))
        diagsquare[0, 0] = kn
        diagsquare[1, 1] = -fn / (self.conf.rconversion * rval) + kt
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
        Hessian[3 * i:(3 * i + 3), 3 * i:(3 * i + 3)] += Hijdiag / mi

        #And once more for the jj contribution, which is the same whizz with the flipped sign of n and t
        # and adjusted A's
        diagsquare = np.zeros((3, 3))
        diagsquare[0, 0] = kn
        diagsquare[1, 1] = -fn / (self.conf.rconversion * rval) + kt
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
        Hessian[3 * j:(3 * j + 3), 3 * j:(3 * j + 3)] += Hjidiag / mj
    return Hessian
    
        
def generateHessianFiles(topDir,numParticles,K_t,K_n,snapShotRange=False):
    
    #First let's collect all the par and int files needed from topDir
    intFileHolder = []
    parFileHolder = []
    for file in os.listdir(topDir):
        if file.startswith("int_"):
            intFileHolder.append(file)
        if file.startswith("par_"):
            parFileHolder.append(file)
            
    intFileHolder = [os.path.join(topDir,file) for file in intFileHolder]
    parFileHolder = [os.path.join(topDir,file) for file in parFileHolder]
    
    if len(intFileHolder)!=len(parFileHolder):
        raise Exception("There are an unequal number of par_ and int_ files in the topDir given.")
    
    
    intStressHolder = np.zeros(len(intFileHolder))
    parStressHolder = np.zeros(len(parFileHolder))
    for i in range(0,len(intFile)):     
        result = re.search('_stress(.*)cl', intFileHolder[i])
        intStressHolder[i] = float(result.group(1))
        result = re.search('_stress(.*)cl', parFileHolder[i])
        parStressHolder[i] = float(result.group(1))
        
        
    #Now that we have all the int and par files we can begin iterating through them to generate the hessian files
    
    for i in range(0,len(intFile)):
        
        currentIntFile = intFileHolder[i]
        currentStress = intStressHolder[i]
        currentParFile = parFileHolder[np.where(parStressHolder==currentStress)]
        contactInfo,radii = hessianDataExtractor(intFile,parFile,numParticles,snapShotRange)
        hessianHolder = np.zeros((3 * numParticles, 3 * numParticles,len(contactInfo)))
        
        for k in range(0,len(contactInfo)):
            hessian = 2DHessianGenerator(contactInfo[k],radii,K_t,K_n)
            hessianHolder[:,:,k] = hessian
        
        
        hessianFileToSave = os.path.join(topDir,"hessian_stress"+str(currentStress)+"cl")
        np.save(hessianFileToSave,hessianHolder)
            
        
        
        
    
    
"""

Check bullshit
"""
        
intFile  = r"D:\data\temp\int_D2N2000VF0.78Bidi1.4_0.5Square_1_pardata_phi0.78_stress100cl.dat"
parFile  = r"D:\data\temp\par_D2N2000VF0.78Bidi1.4_0.5Square_1_pardata_phi0.78_stress100cl.dat"
#Load in the particles radii's (second column), the x positions (third column), and z positions (fifth column).
positionData = np.loadtxt(parFile,usecols=[1,2,4])

#Extract number of snapshots from positionData
numSnapshots = int(np.shape(positionData)[0]/numParticles)


#If the optional variable snapShotRange is not set then process all snapshots.  Otherwise set the correct range.
if snapShotRange == False:
    lowerSnapShotRange = 0
    upperSnapShotRange = numSnapshots
else:
    lowerSnapShotRange = snapShotRange[0]
    upperSnapShotRange = snapShotRange[1]
    if upperSnapShotRange == -1:
        upperSnapShotRange=numSnapshots



#Extract the particle radii's
particleRadii = positionData[:numParticles,0]
#Delete the first column now that we no longer need particle radii
positionData=positionData[:,1:]


# Reshape the particle positions file so that it's a 3D array where each 2D slice is a snapshot.
# For example positionData[:,:,i] would be a 2D array of the x and z positions for the ith snapshot
newPosData = np.zeros((numParticles,2,numSnapshots))

for i in range(0,numSnapshots):
    newPosData[:,:,i] = positionData[i*numParticles:(i+1)*numParticles,:]


positionData = newPosData
positionData = positionData[:,:,lowerSnapShotRange:upperSnapShotRange]
    



#Now lets load in the particle contacts from intFile and ignore the header lines (first 25 lines).
with open(intFile) as f1:
    fileLines = f1.readlines()[24:]

numLines = np.shape(fileLines)[0]

#We'll first find every line in intFile that starts with "#" as that is a line where a new snapshop starts.
counter=0
linesWhereDataStarts=np.array([])
for lines in fileLines:
    if (np.shape(np.fromstring(lines,sep=' '))[0]==0) & ("#" in str(lines)):
        linesWhereDataStarts = np.append(linesWhereDataStarts, counter)
    counter=counter+1

#At this point we can do a sanity check to see if numSnapshots is equal to the number of lines where the data starts.
if np.shape(linesWhereDataStarts)[0]!=numSnapshots:
    raise TypeError("The number of snapshots in the par file does not match the number of snapshots in the int file.  Please make sure both files correspond to the same simulation.")

#Now lets make a python list to store all the different contacts for each snapshot
contactInfo = [0] * (upperSnapShotRange-lowerSnapShotRange)

#Now we'll loop through each snapshot and store only the first three columns.  This should hopefully make this less expensive.
#The first column is the first particle index, the second is the second particle index and the final column tells us the contact type.
#We will also be ignoring any interaction where the contact type is 0 as that is a hydrodynamic interaction.
counter = 0
print("Contact and position data has been converted from the LF_DEM format to the format needed to play the pebble game.  Starting pebble game calculations now!")
for i in range(lowerSnapShotRange,upperSnapShotRange):
    if i==numSnapshots-1:
        #If there is a 0 in the third column then that means the particles are not in contact and we can throw that row our.
        currentContacts = np.genfromtxt(itertools.islice(fileLines, int(linesWhereDataStarts[i]), int(numLines)),usecols=(0, 1, 2,6))
        currentContacts = currentContacts[np.where(currentContacts[:, 2] > 1), :][0]
        if len(currentContacts)==0:
            contactInfo[counter] = np.array([[0],[0]])
        else:    
            contactInfo[counter] = np.concatenate((np.expand_dims(currentContacts[:,0], axis=1),np.expand_dims(currentContacts[:,1], axis=1),np.expand_dims(currentContacts[:,2], axis=1),np.expand_dims(currentContacts[:,3], axis=1)),axis=1)
        
    else:
        currentContacts = np.genfromtxt(itertools.islice(fileLines, int(linesWhereDataStarts[i]), int(linesWhereDataStarts[i + 1])),usecols=(0, 1, 2,6))
        currentContacts = currentContacts[np.where(currentContacts[:, 2] > 1), :][0]
        if len(currentContacts)==0:
            contactInfo[counter] = np.array([[0],[0]])
        else:    
            contactInfo[counter] = np.concatenate((np.expand_dims(currentContacts[:,0], axis=1),np.expand_dims(currentContacts[:,1], axis=1),np.expand_dims(currentContacts[:,2], axis=1),np.expand_dims(currentContacts[:,3], axis=1)),axis=1)
    del currentContacts
    counter=counter+1

#We no longer need fileLines and it takes up a lot of RAM so we can delete it (not sure if this is needed, python interpreters are pretty good about this stuff)
del fileLines




snapShot = 4
contactNumber =143
currentContact = contactInfo[snapShot][contactNumber]


#Now look at a contact
i=int(currentContact[0])
j=int(currentContact[1])
s = currentContact[3]

R_i = particleRadii[i]
R_j = particleRadii[j]
posI = positionData[i,:,snapShot]
posJ = positionData[j,:,snapShot]

sep  = np.sqrt(np.sum((posI-posJ)**2))

overlap = R_i+R_j-sep

S = 2*sep/(R_i+R_j)-2

print("overlap = "+str(overlap))
print("s = "+str(s))
print("S = "+str(S))
