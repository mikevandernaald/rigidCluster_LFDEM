import numpy as np
import sys
import os
from matplotlib import pyplot
import Configuration_LFDEM as CF #Mike:  I modified the Configuration class so we don't have to load in using text files.
import Pebbles as PB
import Hessian as HS
import Analysis as AN
import itertools
'''
Date:  2/5/2021
Authors:  Kenan Tang, Toka Eid, Mike van der Naald
This is a collection of functions that allow us to use Silke's rigid cluster algorithms to process suspension simulations 
from LF_DEM.  

Link to LF_DEM Github:

Link to Silke's Rigid Cluster Github:

To run on the cluster just use the following code to point the Python interpreter to where you keep this code:
import sys
sys.path.append('/home/mrv/pebbleCode/rigidCluster_LFDEM')

'''


def pebbleGame_LFDEMSnapshot(dataFile,parFile,intFile,outputDir=False,snapShotRange=False):
    """
    This program takes in the path to the data_, int_, and par_ files from LF_DEM simulation output and then feeds them
    into a code that identifies rigid cluster statistics from each snapshot in the simulation.  These statistics are then
    outputted into a textfile in outputDir and are also returned as a list via the function call return.  If you only
    want to process some of the snapshots you can put that in the variable snapShotRange.
    :param dataFile: This is path to the data_ file outputted from the LF_DEM simulations.
    :param parFile:  This is path to the int_ file outputted from the LF_DEM simulations.
    :param intFile:  This is path to the par_ file outputted from the LF_DEM simulations.
    :param outputDir:  This is the directory where a textfile is outputted with the cluster statistics for each snapshot.
    :param snapShotRange:  This is the range of snapshots you want to calculate cluster statistics for.
    Ex. if snapShotRange=[0,5] then the program would calcualte cluster statistics for the first 5 snapshots
    :return:
    """

    #Open the int file and read the number of particles from the second line and the system size from the fourth line
    with open(intFile) as fp:
        for i, line in enumerate(fp):
            if i==1:
                res = [int(i) for i in line.split() if i.isdigit()]
                numParticles=res[0] #This skips the first five characters in the line since they're always "# np "
            if i==3:
                systemSize = float(line[5:]) #This skips the first five characters in the line since they're always "# Lx "

    #Now we need to get the strain rate for each snapshot.  These are stored as the third colum in the dataFile.
    strainRates = np.loadtxt(dataFile,usecols=[2])

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
    positionData = positionData.reshape(numParticles,2,numSnapshots)

    #Now lets load in the particle contacts from intFile and ignore the header lines (first 25 lines).
    with open(intFile) as f1:
        fileLines = f1.readlines()[25:]

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
    for i in range(lowerSnapShotRange,upperSnapShotRange):
        if i==numSnapshots-1:
            #If there is a 0 in the third column then that means the particles are not in contact and we can throw that row our.
            currentContacts = np.genfromtxt(itertools.islice(fileLines, int(linesWhereDataStarts[i]), int(numLines)),usecols=(0, 1, 2))
            contactInfo[counter] = currentContacts[np.where(currentContacts[:, 2] != 0), :]
        else:
            currentContacts = np.genfromtxt(itertools.islice(fileLines, int(linesWhereDataStarts[i]), int(linesWhereDataStarts[i + 1])),usecols=(0,1,2))
            contactInfo[counter] = currentContacts[np.where(currentContacts[:, 2] != 0), :]
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
        ThisConf = CF.Configuration(numParticles,systemSize,strainRates[i],particleRadii)
        ThisConf.readSimdata(positionData[:,:,i],currentContactData[0,:,:],i)
        ThisPebble = PB.Pebbles(ThisConf, 3, 3, 'nothing', False)
        ThisPebble.play_game()

        cidx, clusterall, clusterallBonds, clusteridx, BigCluster = ThisPebble.rigid_cluster()
        clusterHolder[counter] = [cidx, clusterall, clusterallBonds, clusteridx, BigCluster]
        counter=counter+1

    return clusterHolder








