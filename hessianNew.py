# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 09:45:34 2022

@author: mikev
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


def hessianDataExtractor(intFile,parFile,inputFile,numParticles,snapShotRange=False):
    
    #let's first get the spring constants from the input files
    fp = open(inputFile)
    for i, line in enumerate(fp):
        if "kn =" in line:
            kn = line
        elif "kt =" in line:
            kt = line
            break
    fp.close()
    
    result = re.search('kt = (.*)cl;', kt)
    kt = float(result.group(1))
    
    result = re.search('kn = (.*)cl;', kn)
    kn = float(result.group(1))

    
    radii = np.loadtxt(parFile,usecols=[1])
    #Extract number of snapshots from radiiData before discarding most of it
    numSnapshots = int(np.shape(radii)[0]/numParticles)

    radii = radii[:numParticles]

    
    
    if snapShotRange == False:
        lowerSnapShotRange = 0
        upperSnapShotRange = numSnapshots
    else:
        lowerSnapShotRange = snapShotRange[0]
        upperSnapShotRange = snapShotRange[1]
    
    #Let's first extract the contact information by opening up the int file.
    #Ignore the header lines (first 25 lines).
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
                contactInfo[counter] = np.array([[0],[0]])
            else:    
                contactInfo[counter] = np.concatenate((np.expand_dims(currentContacts[:,0], axis=1),np.expand_dims(currentContacts[:,1], axis=1),np.expand_dims(currentContacts[:,2], axis=1)),axis=1)
            
        else:
            currentContacts = np.genfromtxt(itertools.islice(fileLines, int(linesWhereDataStarts[i]), int(linesWhereDataStarts[i + 1])),usecols=(0,1,2,3,5,6))
            currentContacts = currentContacts[np.where(currentContacts[:, 2] > 1), :][0]
            if len(currentContacts)==0:
                contactInfo[counter] = np.array([[0],[0]])
            else:    
                contactInfo[counter] = np.concatenate((np.expand_dims(currentContacts[:,0], axis=1),np.expand_dims(currentContacts[:,1], axis=1),np.expand_dims(currentContacts[:,2], axis=1)),axis=1)
        del currentContacts
        counter=counter+1

    #We no longer need fileLines and it takes up a lot of RAM so we can delete it (not sure if this is needed, python interpreters are pretty good about this stuff)
    del fileLines


    return contactInfo,radii


def 2DHessianGenerator(contactInformation,positions,radii,K_t,K_n,particleDensity=1,cylinderHeight=1):
    """
    contactInformation has a row for each contact with the first column being the 
    ID of the first particle, the second column being the ID of the second
    particle, the third column being the n_x component of the normal force, 
    the fourth column being the n_y component of the normal force, 
    the fifth column determines whether the contacat is sliding or not, finally the
    last column is the overlap between two particles
    """
    numParticles = len(radii)
    Hessian = np.zeros((3 * numParticles, 3 * numParticles))
    
    
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
