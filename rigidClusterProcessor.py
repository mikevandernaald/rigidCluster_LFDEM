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
sys.setrecursionlimit(1500000)

"""
Date:  2/5/2021
Authors: Mike van der Naald
This is a collection of functions that allow us to use Silke's rigid cluster algorithms to process suspension simulations 
from LF_DEM.  

Link to LF_DEM Github: https://github.com/ryseto/LF_DEM

Link to Silke's Rigid Cluster Github: https://github.com/silkehenkes/RigidLibrary


NOTE:  IN IT'S CURRENT FORM THESE FUNCTIONS ONLY WORK ON TWO DIMENSIONAL LF_DEM
DATA THAT USES THE CRITICAL LOAD MODEL (I.E. HYDRODYNAMICS+REPULSIVE CONTACT FORCE+COULOMBIC FRICTION)


There are three essential functions that will allow one to calulate rigid cluster statistics from 2D LF_DEM data.


The first function "pebbleGame_LFDEMSnapshot" takes in a parFile and an intFile from LF_DEM data and outputs 
outputs cluster sizes, number of bonds in a cluster, and optionally all of the particle IDS in each cluster 
as well as the Pebble object that Silke's code uses.'

The second function is "rigFileGenerator" which takes in a topDir where it looks for par_ and int_ files to process. 
It also takes "outputDir" which is the directory where it dumps the processed rig_ files.  The optional argument snapShotRange
determines which simulation snapshots are processed.  The optional argument "reportIDs" is set to True meaning that the rig_ files
will have the particle IDs for each cluster, setting this to False will increase the runtime substantially and is useful if you only care about 
cluster size statistics.  The optional argument "stressControlled" sets whether or not the function will look for stress controlled simulations (True)
or rate controlled simulations (False)

The third function "rigFileReader" reads in a particular rigFile given the path to the output rigFile.  snapShotRange sets which snapshots are read from the 
rig file with the default being all snap shots.  Finally readInIDs sets whether or not the function will try to read in the particle IDs from the rig file which 
increases the runtime quite a bit.

The remaining functions are just helper functions that I (mike van der Naald) cobbled together to analyze and process rig_ files or other LF_DEM data.  If one 
looks interesting to you go ahead and email me (mikevandernaald@gmail.com) and I can try to add some comments to it.

To run on the cluster just use the following code to point the Python interpreter to where you keep this code:
import sys
sys.path.append('/home/mrv/pebbleCode/rigidCluster_LFDEM')
"""



def pebbleGame_LFDEMSnapshot(parFile,intFile,snapShotRange=False,returnPebbleIDandObj = True):
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
    
    
    def rigidClusterDataGenerator(pebbleObj,returnClusterIDs=True):
        
        """
        This is a helper function that returns the rigid cluster properties from a pebbleObj which is constructed later in the function.
        """
    
        #Load in all the relevant data.  The first column has the ID of the cluster and the second and third rows tell you the particles i and j which are in that cluster ID
        clusterIDHolder = np.transpose(np.vstack([pebbleObj.cluster,pebbleObj.Ifull,pebbleObj.Jfull]))
        
        
        #Remove all rows that have -1 in the first column.  Those are contacts that are not participating in a cluster
        clusterIDHolder = clusterIDHolder[clusterIDHolder[:,0] != -1]
        
        numClusters = len(np.unique(clusterIDHolder[:,0]))
        
        clusterSizes = np.zeros(numClusters)
        numBondsPerCluster = np.zeros(numClusters)
        if returnClusterIDs == True:
            clusterID = [0]*numClusters
        
        counter = 0
        for i in np.unique(clusterIDHolder[:,0]):
            currentCluster = clusterIDHolder[clusterIDHolder[:,0]==i][:,1:]
            
            currentCluster = np.unique(np.sort(currentCluster,axis=1), axis=0)
        
            (numBonds,_) = np.shape(currentCluster)
            
            
            numBondsPerCluster[counter] = numBonds
            clusterSizes[counter] = len(np.unique(currentCluster.flatten()))
            if len(np.unique(currentCluster.flatten())) == 0:
                breakpoint()
            if returnClusterIDs == True:
                clusterID[counter] = currentCluster
            
            counter=counter+1
        if returnClusterIDs == True:
            return clusterSizes,numBondsPerCluster,clusterID
        else:
            return clusterSizes,numBondsPerCluster
    

    
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
            ThisPebble.rigid_cluster()
            if returnPebbleIDandObj == False:
                clusterSizes,numBondsPerCluster = rigidClusterDataGenerator(ThisPebble,False)
            else:
                clusterSizes,numBondsPerCluster,clusterID = rigidClusterDataGenerator(ThisPebble)
            if np.sum(clusterSizes)==0:
                clusterHolder[counter]=[0]
            else:
                if returnPebbleIDandObj == True:
                    clusterHolder[counter] = [ clusterSizes, numBondsPerCluster,clusterID,ThisPebble]
                else:
                    clusterHolder[counter] = [ clusterSizes, numBondsPerCluster]
        counter=counter+1
    return clusterHolder

def rigFileGenerator(topDir,outputDir,snapShotRange=False,reportIDS=True,stressControlled=True,stressRange=False):
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
        if stressControlled==True:
            result = re.search('_stress(.*)cl', currentFile)
            currentStress = result.group(1)
            correspondingIntFile = [i for i in intFiles if '_stress'+currentStress+'cl' in i]
        else:
            result = re.search('_rate(.*)cl', currentFile)
            currentRate = result.group(1)
            correspondingIntFile = [i for i in intFiles if '_rate'+currentRate+'cl' in i]
                
        currentIntFile = os.path.join(topDir,correspondingIntFile[0])
        currentParFile = os.path.join(topDir,currentFile)
        
        
        if stressRange!=False:
            if stressRange[0] <= float(currentStress) <= stressRange[1]:
                print("We're currently processing the stress: " + currentStress )
            else:
                break
            
        
        currentClusterInfo = pebbleGame_LFDEMSnapshot(currentParFile,currentIntFile,snapShotRange)
        
        
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
                        fp.write(str(int(j))+'\t')
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
                    for j in currentSnapShot[1]:
                        fp.write(str(int(j))+'\t')
                    fp.write('\n')
            fp.write('\n')
        fp.close()
        if reportIDS==True: 
            with open(rigidClusterFileName, 'a') as fp:
                fp.write('#Rigid Cluster IDs \n')
                
                for i in range(0,len(currentClusterInfo)):
                    
                    fp.write('#snapShot = '+str(i)+'\n')
                    
                    currentSnapShot = currentClusterInfo[i]
                    
                    if currentSnapShot==[0]:
                        fp.write(str(0)+'\n')
                    else:
                        numClusters = len(currentSnapShot[0])
                        
                        for k in range(0,numClusters):
                            currentTuplesToSave = currentSnapShot[2][k].flatten()
                            for j in range(0,len(currentTuplesToSave)):
                                if j==len(currentTuplesToSave)-1:
                                    fp.write(str(int(currentTuplesToSave[j]))+"\n")
                                else:
                                    fp.write(str(int(currentTuplesToSave[j]))+",")
                            
            
            
            fp.close()            
    
def rigFileReader(rigFile,snapShotRange=False,readInIDS=True):
    
    
    with open(rigFile, "r") as f1:   
        fileLines = f1.readlines()
        
    indexOfDataSplit = fileLines.index('#Rigid Cluster Bond Numbers\n')
    numSnapshots = indexOfDataSplit-2

    
    
    #If the optional variable snapShotRange is not set then process all snapshots.  Otherwise set the correct range.
    if snapShotRange == False:
        lowerSnapShotRange = 0
        upperSnapShotRange = numSnapshots
    else:
        if snapShotRange[1]==-1:
            lowerSnapShotRange = snapShotRange[0]
            upperSnapShotRange = numSnapshots
        else:
            lowerSnapShotRange = snapShotRange[0]
            upperSnapShotRange = snapShotRange[1]
            
    
    
    rigidClusterSizes = [0]*(upperSnapShotRange-lowerSnapShotRange)
    numBonds = [0]*(upperSnapShotRange-lowerSnapShotRange)
    
    counter=0
    for i in range(lowerSnapShotRange,upperSnapShotRange):
        
        rigidClusterSizes[counter] = np.fromstring(fileLines[i+1].replace('\t',' ').replace(',', '').replace('\n',''),sep=' ')
        numBonds[counter] = np.fromstring(fileLines[i+1+indexOfDataSplit].replace('\t',' ').replace(',', '').replace('\n',''),sep=' ')
        counter=counter+1
        
    if readInIDS == True:
        clusterIDs = [0]*(upperSnapShotRange-lowerSnapShotRange)
        
        
        snapShotLineIndices = np.zeros(numSnapshots)
        #First we'll find the indices where each snapshot starts
        for i in range(0,numSnapshots):
            snapShotLineIndices[i] = int(fileLines.index('#snapShot = '+str(i)+'\n'))
        
        snapShotStartingPoints = snapShotLineIndices + 1
        snapShotEndingPoints = snapShotLineIndices-1
        snapShotEndingPoints = np.append(snapShotEndingPoints[1:],len(fileLines))
        
        counter=0
        for i in range(lowerSnapShotRange,upperSnapShotRange):
            currentFileLines = fileLines[int(snapShotStartingPoints[i]):int(snapShotEndingPoints[i])+1]

            clusterIDHolder = []
            for lines in currentFileLines:
                currentLineArray = np.fromstring(lines.replace('\n',''),sep=',')
                if np.array_equal(currentLineArray,np.array([0])):
                    clusterIDHolder.append(currentLineArray)
                else:
                    currentLineArray = currentLineArray.reshape( ( int(len(currentLineArray)/2), 2) )
                    clusterIDHolder.append(currentLineArray)
            
            clusterIDs[counter] = clusterIDHolder
            counter=counter+1
        return (rigidClusterSizes,numBonds,clusterIDs)
    else:
        return rigidClusterSizes,numBonds
    
def viscosityAverager(dataFile):
    #dataFile = r"C:\Users\mikev\Documents\data\rigidClusterData\data_D2N2000VF0.8Bidi1.4_0.5Square_1_pardata_phi0.8_stress100cl.dat"
    
    
    with open(dataFile, "r") as f1:   
        fileLines = f1.readlines()
        
    totalNumLines = len(fileLines)
    
    viscosityHolder = np.zeros(totalNumLines-45)
    
    counter=0
    for i in range(45,totalNumLines):
        np.fromstring(fileLines[i].replace("n", "0"),sep=' ')
        viscosityHolder[counter] = np.fromstring(fileLines[i].replace("n", "0"),sep=' ')[4]
          
        counter=counter+1
    return viscosityHolder
         
def rigidClusterLength(rigFile,parFile,numParticles,Lx,Ly,snapShotRange=False,rotatePositions=False):
    
    #Load in position data
    positionData = np.loadtxt(parFile,usecols=[1,2,4])
    numSnapshots = int(np.shape(positionData)[0]/numParticles)
    
    
    if snapShotRange == False:
        lowerSnapShotRange = 0
        upperSnapShotRange = numSnapshots
    else:
        if snapShotRange[1]==-1:
            lowerSnapShotRange = snapShotRange[0]
            upperSnapShotRange = numSnapshots
        else:
            lowerSnapShotRange = snapShotRange[0]
            upperSnapShotRange = snapShotRange[1]
            
    

    def snapShotLengthCalc(pos,Lx,Ly):
        (numParticles,_) = np.shape(pos)
        
        
        #x distances 
        pos_1d = pos[:, 0][:, np.newaxis] # shape (N, 1)
        dist_1dx = pdist(pos_1d)  # shape (N * (N - 1) // 2, )
        dist_1dx[dist_1dx > Lx/2] -= Lx
        
        #y distances next
        pos_1d = pos[:, 1][:, np.newaxis]  # shape (N, 1)
        dist_1dy = pdist(pos_1d)  # shape (N * (N - 1) // 2, )
        dist_1dy[dist_1dy > Ly/2] -= Ly
        
        
        largestExtentX = np.max(dist_1dx)
        largestExtentY = np.max(dist_1dy)
        largestTotalExtent = np.max(np.sqrt(dist_1dx**2+dist_1dy**2))
        
        return largestExtentX,largestExtentY,largestTotalExtent
    
    
    #Read in rigid cluster information
    _,_,clusterIDs = rigFileReader(rigFile,snapShotRange)
    
    #Delete the first column now that we no longer need particle radii
    positionData=positionData[:,1:]
    
    #If we're rotating positions this is where we'll do it
    if rotatePositions != False:
        rotationMatrix = (1/np.sqrt(2))*np.array([[1,1],[-1,1]])
        
        numRows,_ = np.shape(positionData)
        
        for i in range(0,numRows):
            positionData[i,:] = np.matmul(rotationMatrix,positionData[i,:])
        
    

    
    #Reformat the position data to be a 3 dimensional array
    newPosData = np.zeros((numParticles,2,numSnapshots))
    
    for i in range(0,numSnapshots):
        newPosData[:,:,i] = positionData[i*numParticles:(i+1)*numParticles,:]

    positionData = newPosData
    positionData = positionData[:,:,lowerSnapShotRange:upperSnapShotRange]
    
    
    xExtentHolder = (len(clusterIDs))*[[]]
    yExtentHolder = (len(clusterIDs))*[[]]
    totalExtentHolder = (len(clusterIDs))*[[]]

    for i in range(0,len(clusterIDs)):
        print(i)
        currentPos = positionData[:,:,i]
        currentClusterIDs = clusterIDs[i]
        
        checker=0
        for clusters in currentClusterIDs:
            checker = np.sum(clusters)
        
        
        
        if checker!=0:
            clusterExtentHolderX = []
            clusterExtentHolderY = []
            clusterExtentHolderTotal=[]
            
            for clusters in currentClusterIDs:
    
                allIDsInCurrentCluster = np.unique(clusters)
                allIDsInCurrentCluster = [int(ids) for ids in allIDsInCurrentCluster]
                positionsOfParticlesInCurrentCluster = currentPos[allIDsInCurrentCluster,:]
                
                (largestExtentX,largestExtentY,totalExtent) = snapShotLengthCalc(positionsOfParticlesInCurrentCluster,Lx,Ly)
                
                clusterExtentHolderX.append(largestExtentX)
                clusterExtentHolderY.append(largestExtentY)
                clusterExtentHolderTotal.append(totalExtent)
    
                
            xExtentHolder[i] = np.max(clusterExtentHolderX)
            yExtentHolder[i] = np.max(clusterExtentHolderY)
            totalExtentHolder[i] = np.max(clusterExtentHolderY)
        else:
            xExtentHolder[i] = 0
            yExtentHolder[i] = 0
            totalExtentHolder[i] = 0
            
    return (xExtentHolder,yExtentHolder,totalExtentHolder)
        
def largestClusterCalc(rigFile,snapShotStartingPoint,maximum=True):
    
    (rigidClusterSizes,numBonds,clusterIDs) = rigFileReader(rigFile,[snapShotStartingPoint,-1])
    
    largestClusters = np.zeros(len(rigidClusterSizes))
    
    counter=0
    for currentClusterList in rigidClusterSizes:
        if maximum==True:
            largestClusters[counter] = np.max(currentClusterList)
        else:
            largestClusters[counter] = currentClusterList
        counter=counter+1
    
    #If the value in the list is true then it is larger than the threshold size.
    
    return largestClusters

def allClusterCalc(rigFile,snapShotStartingPoint):
    
    (rigidClusterSizes,numBonds,clusterIDs) = rigFileReader(rigFile,[snapShotStartingPoint,-1])
    
    allClusters = np.array([])
    
    for currentClusterList in rigidClusterSizes:
        allClusters = np.concatenate((allClusters,currentClusterList))

    return allClusters

def phiRigCalc(topDir,sizeThreshold,percentageSnapShotThreshold):
    
    #helper functions
    
    def stepFunction(x):
        if x >= sizeThreshold:
            return 1
        else:
            return 0
    
    
    listOfDirs = [os.path.join(topDir, o) for o in os.listdir(topDir) if os.path.isdir(os.path.join(topDir,o))]
    listOfPhi = [float(listOfDirs[i][-4:].replace("F", "").replace("V", "")  ) for i in range(0,len(listOfDirs))]

    
    percentageHolder = np.zeros(len(listOfPhi))
    
    #Now let's calculate the largest cluster for each phi
    counter=0
    for currentDir in listOfDirs:
        #First we need to find the rig file for the highest stress
        listOfFiles = [f for f in listdir(currentDir) if isfile(join(currentDir, f))]
        listOfFiles = [f for f in listOfFiles if "stress100cl" in f]
        currentRigFile = [f for f in listOfFiles if "rig_" in f][0]
        
        currentMaxClusterSizes = largestClusterCalc(os.path.join(currentDir,currentRigFile),50)
        
        if len(currentMaxClusterSizes)!=0:
        
            snapShotsAboveThreshold = [stepFunction(x) for x in currentMaxClusterSizes]
                    
            percentageHolder[counter] = (1/len(snapShotsAboveThreshold))*np.sum(snapShotsAboveThreshold)
        else:
            percentageHolder[counter] =np.nan
            
        counter=counter+1
    
    
    percentageHolder = percentageHolder >= percentageSnapShotThreshold
    
    phiWithPercentage = np.vstack((percentageHolder,listOfPhi))
    
    
    
    if np.sum(phiWithPercentage[0,:]==1) ==0:
        return np.nan
    else: 
        phiRig = listOfPhi[np.min(np.where(phiWithPercentage[0,:]==1))]
        return phiRig
     
def phiRigCalcFuncStress(topDir,sizeThreshold,percentageSnapShotThreshold,snapShotStartingPoint):
    
    #helper functions
    def stepFunction(x):
        if x >= sizeThreshold:
            return 1
        else:
            return 0
    
    rigFiles =  [f for f in listdir(topDir) if isfile(join(topDir, f)) and "rig_" in f]
    rigFiles = [os.path.join(topDir,file) for file in rigFiles]
    
    #The firsrt column is whether or not the max cluster was found and second column is stress and t
    percentageHolder = np.zeros((len(rigFiles),3))
    
    #Now lets calcualte phiRig for each stress
    counter=0
    for currentRigFile in rigFiles:
        
        currentStress = float(re.search('_stress(.*)cl', currentRigFile).group(1))
        percentageHolder[counter,0] = currentStress
        
        currentMaxClusterSizes = largestClusterCalc(currentRigFile,snapShotStartingPoint)
        percentageHolder[counter,2] = len(currentMaxClusterSizes)
        print(len(currentMaxClusterSizes))
        
        if len(currentMaxClusterSizes)!=0:
            snapShotsAboveThreshold = [stepFunction(x) for x in currentMaxClusterSizes]
                    
            percentageHolder[counter,1] = (1/len(snapShotsAboveThreshold))*np.sum(snapShotsAboveThreshold) >= percentageSnapShotThreshold
        else:
            percentageHolder[counter,1] =np.nan
            
        counter=counter+1
        
    return np.sort(percentageHolder,0)

def intFileStrainReader(intFile):
    
    
    #Now lets load in the particle contacts from intFile and ignore the header lines (first 25 lines).
    with open(intFile) as f1:
        fileLines = f1.readlines()[24:]
    
    strainList = []
    for lines in fileLines:
        if "#" in lines:
            floats_list = []
            for item in str(lines[1:]).split():
                floats_list.append(float(item))
            strainList.append(floats_list[2])
            
    return np.array(strainList)



            
        
    