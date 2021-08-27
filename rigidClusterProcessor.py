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
"""
Date:  2/5/2021
Authors: Mike van der Naald
This is a collection of functions that allow us to use Silke's rigid cluster algorithms to process suspension simulations 
from LF_DEM.  

Link to LF_DEM Github: https://github.com/ryseto/LF_DEM

Link to Silke's Rigid Cluster Github: https://github.com/silkehenkes/RigidLibrary

To run on the cluster just use the following code to point the Python interpreter to where you keep this code:
import sys
sys.path.append('/home/mrv/pebbleCode/rigidCluster_LFDEM')
"""



def pebbleGame_LFDEMSnapshot(parFile,intFile,snapShotRange=False):
    """
    This program takes in the path to the data_, int_, and par_ files from LF_DEM simulation output and then feeds them
    into a code that identifies rigid cluster statistics from each snapshot in the simulation.  These statistics are then
    returned as a list.  If you only want to process some of the snapshots you can put that in the variable snapShotRange.
    :param parFile:  This is path to the int_ file outputted from the LF_DEM simulations.
    :param intFile:  This is path to the par_ file outputted from the LF_DEM simulations.
    :param snapShotRange:  This is the range of snapshots you want to calculate cluster statistics for.
    Ex. if snapShotRange=[0,5] then the program would calculate cluster statistics for the first 5 snapshots

    :return:


    """
    
        
    
    with open(intFile) as fp:
        for i, line in enumerate(fp):
            if i==1:
                res = [int(i) for i in line.split() if i.isdigit()]
                numParticles=res[0] #This skips the first five characters in the line since they're always "# np "
            if i==3:
                systemSizeLx = float(line[5:]) #This skips the first five characters in the line since they're always "# Lx "
            if i==5:
                systemSizeLz = float(line[5:]) #This skips the first five characters in the line since they're always "# Lx "
    
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
            currentContacts = np.genfromtxt(itertools.islice(fileLines, int(linesWhereDataStarts[i]), int(numLines)),usecols=(0, 1, 2))
            currentContacts = currentContacts[np.where(currentContacts[:, 2] > 1), :][0]
            if len(currentContacts)==0:
                contactInfo[counter] = np.array([[0],[0]])
            else:    
                contactInfo[counter] = np.concatenate((np.expand_dims(currentContacts[:,0], axis=1),np.expand_dims(currentContacts[:,1], axis=1),np.expand_dims(currentContacts[:,2], axis=1)),axis=1)
            
        else:
            currentContacts = np.genfromtxt(itertools.islice(fileLines, int(linesWhereDataStarts[i]), int(linesWhereDataStarts[i + 1])),usecols=(0, 1, 2))
            currentContacts = currentContacts[np.where(currentContacts[:, 2] > 1), :][0]
            if len(currentContacts)==0:
                contactInfo[counter] = np.array([[0],[0]])
            else:    
                contactInfo[counter] = np.concatenate((np.expand_dims(currentContacts[:,0], axis=1),np.expand_dims(currentContacts[:,1], axis=1),np.expand_dims(currentContacts[:,2], axis=1)),axis=1)
        del currentContacts
        counter=counter+1

    #We no longer need fileLines and it takes up a lot of RAM so we can delete it (not sure if this is needed, python interpreters are pretty good about this stuff)
    del fileLines
    #Now that we have all the particle position information and contact information for each snap shot we can loop over each
    # #snapshot and play the PebbleGame on it.

    #This list will hold all the cluster information so let's inialize it with zeros.
    clusterHolder = [0] * (upperSnapShotRange-lowerSnapShotRange)
    counter=0
    for i in range(lowerSnapShotRange,upperSnapShotRange):
        
        currentContactData = contactInfo[counter]
        
        ThisConf = CF.Configuration(numParticles,systemSizeLx,0,particleRadii)

        if np.array_equal(currentContactData,np.array([[0],[0]])):
            clusterHolder[counter]=[0]
        else:
            ThisConf.readSimdata(positionData[:,:,counter],currentContactData,i)
            ThisPebble = PB.Pebbles(ThisConf, 3, 3, 'nothing', False)
            ThisPebble.play_game()
            
            _, clusterall, clusterallBonds, _, _ = ThisPebble.rigid_cluster()
            if clusterall==[]:
                clusterHolder[counter]=[0]
            else:
                clusterHolder[counter] = [ clusterall, clusterallBonds,ThisPebble,ThisConf]
        counter=counter+1

    return clusterHolder



def rigFileGenerator(topDir,outputDir):
    """
    This finds all par and int files in a directory and spits out their rigidcluster statistics
    into a rig_ file
    """
    
    
    parFiles = []
    intFiles = []
    

    for file in os.listdir(topDir):
        if "int_" in file:
            intFiles.append(file)
        if "par_" in file:
            parFiles.append(file)
            
            
    for currentFile in parFiles:
        result = re.search('_stress(.*)cl', currentFile)
        currentStress = result.group(1)
        
        
        
        correspondingIntFile = [i for i in intFiles if '_stress'+currentStress+'cl' in i]
        
        currentIntFile = os.path.join(topDir,correspondingIntFile[0])
        currentParFile = os.path.join(topDir,currentFile)
        
        
        currentClusterInfo = pebbleGame_LFDEMSnapshot(currentParFile,currentIntFile)
        
        
        result = re.search('par_(.*).dat', currentFile)
        currentFileName = result.group(1)
        
        rigidClusterFileName = os.path.join(topDir,"rig_"+currentFileName+".dat")
        
        with open(rigidClusterFileName, 'w') as fp:
            fp.write('#Rigid Cluster Sizes \n')
            
            for i in range(0,len(currentClusterInfo)):
                
                currentSnapShot = currentClusterInfo[i]
                
                if currentSnapShot==[0]:
                    fp.write(str(0)+'\n')
                else:
                    for j in currentSnapShot[0]:
                        fp.write(str(j)+'\t')
                    fp.write('\n')
                    
            fp.write('\n')
            fp.write('\n')
            fp.write('#Rigid Cluster Bond Numbers\n')
            fp.close()
            
            with open(rigidClusterFileName, 'a') as fp:
                
                for i in range(0,len(currentClusterInfo)):
                    
                    currentSnapShot = currentClusterInfo[i]
                    
                    if currentSnapShot==[0]:
                        fp.write(str(0)+'\n')
                    else:
                        for j in currentSnapShot[0]:
                            fp.write(str(j)+'\t')
                        fp.write('\n')
            
            fp.close()
            
            
    
    
    
    
    
    
