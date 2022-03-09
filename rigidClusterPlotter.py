# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 09:55:13 2021

@author: mikev
"""

import svgwrite
import numpy as np
import itertools
import os 
import rigidClusterProcessor
import Hessian as HS
import Analysis as AN
import matplotlib
import re
import matplotlib.ticker as mticker
from PIL import Image
sys.setrecursionlimit(15000000000000)



def dataExtractorLFDEMPositionsRadii(parFile,intFile,snapShotRange=False):
    
    
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
        if snapShotRange[1]==-1:
            lowerSnapShotRange = snapShotRange[0]
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
    
    return positionData,particleRadii,systemSizeLx,systemSizeLz,snapShotRange

def dataExtractorLFDEMAllForces(parFile,intFile,snapShotRange=False):
    
    
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
        if snapShotRange[1]==-1:
            lowerSnapShotRange = snapShotRange[0]
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
    contactInfoFrictional = [0] * (upperSnapShotRange-lowerSnapShotRange)
    contactInfoHydro = [0] * (upperSnapShotRange-lowerSnapShotRange)
    contactInfoFrictionless = [0] * (upperSnapShotRange-lowerSnapShotRange)
    
    #Now we'll loop through each snapshot and store only the first three columns.  This should hopefully make this less expensive.
    #The first column is the first particle index, the second is the second particle index and the final column tells us the contact type.
    #We will also be ignoring any interaction where the contact type is 0 as that is a hydrodynamic interaction.
    counter = 0
    for i in range(lowerSnapShotRange,upperSnapShotRange):
        if i==numSnapshots-1:
            currentContacts = np.genfromtxt(itertools.islice(fileLines, int(linesWhereDataStarts[i]), int(numLines)),usecols=(0,1,2,7,11))
        else:
            currentContacts = np.genfromtxt(itertools.islice(fileLines, int(linesWhereDataStarts[i]), int(linesWhereDataStarts[i + 1])),usecols=(0,1,2,7,11))
        
        
        np.ones(len(currentContacts[:,2]))*(currentContacts[:,2]==1) 
        
        magnitudeOfHydroForces = currentContacts[:,3]
        magnitudeOfFrictionalForces = currentContacts[:,4]*np.ones(len(currentContacts[:,2]))*(currentContacts[:,2]>1) 
        magnitudeOfFrictionlessForces = currentContacts[:,4]*np.ones(len(currentContacts[:,2]))*(currentContacts[:,2]==1) 
        
        
            
        if len(currentContacts)==0:
            contactInfoFrictional[counter] = 0
            contactInfoHydro[counter] = 0
            contactInfoFrictionless[counter] = 0
        else:
            contactInfoFrictional[counter] = np.concatenate((np.expand_dims(currentContacts[:,0], axis=1),np.expand_dims(currentContacts[:,1], axis=1),np.expand_dims(magnitudeOfFrictionalForces, axis=1)),axis=1)
            contactInfoFrictionless[counter] = np.concatenate((np.expand_dims(currentContacts[:,0], axis=1),np.expand_dims(currentContacts[:,1], axis=1),np.expand_dims(magnitudeOfFrictionlessForces, axis=1)),axis=1)
            contactInfoHydro[counter] = np.concatenate((np.expand_dims(currentContacts[:,0], axis=1),np.expand_dims(currentContacts[:,1], axis=1),np.expand_dims(magnitudeOfHydroForces, axis=1)),axis=1)
        del currentContacts
        counter=counter+1
    
    return (contactInfoHydro,contactInfoFrictional,contactInfoFrictionless,positionData,particleRadii,systemSizeLx,systemSizeLz,numParticles)

def rigidClusterPlotGenerator(fileName,snapShot,parFile,intFile,rigFile,rigidClusterStrokeWidth):
    colorForCluster = matplotlib.colors.to_hex([0.085, 0.532, 0.201])

    #Load in the positions and radii and plot the packing
    (currentPosData,particleRadii,systemSizeLx,systemSizeLz,_) = dataExtractorLFDEMPositionsRadii(parFile,intFile,[snapShot,snapShot+1])

    dwg = svgwrite.Drawing(size=(systemSizeLx+systemSizeLx/10, systemSizeLz+systemSizeLz/10))
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), rx=None, ry=None, fill='rgb(255,255,255)'))
    
    currentPosData = currentPosData[:,:,0]
    
    xPos = currentPosData[:,0]+ 11 * systemSizeLx / 20
    yPos = currentPosData[:,1]+ 11 * systemSizeLz / 20
    #First plot all the circles
    numParticles = len(particleRadii)
    for i in range(0,numParticles):
        dwg.add(svgwrite.shapes.Circle(center=(xPos[i], yPos[i]), r=particleRadii[i],fill='white',stroke='black',stroke_width=.1,))

    

    #Load in the rigidCluster IDs for the packing
    (_,_,clusterIDs)=rigidClusterProcessor.rigFileReader(rigFile,[snapShot,snapShot+1])

    #Loop through all the clusters and plot them on the previous SVG object
    for clusters in clusterIDs[0]:
        if len(clusters)==1:
            print("fuck")
        else:
            numContactsInCluster = np.shape(clusters)[0]
            
            for i in range(0,numContactsInCluster):
                particleI = int(clusters[i,0])
                particleJ = int(clusters[i,1])
                
                x0 = currentPosData[particleI,0] + 11 * systemSizeLx / 20
                y0 = currentPosData[particleI,1] + 11 * systemSizeLz / 20
                
                x1 = currentPosData[particleJ,0] + 11 * systemSizeLx / 20
                y1 = currentPosData[particleJ,1] + 11 * systemSizeLz / 20
                if (x0-x1)**2 + (y0-y1)**2 < systemSizeLx/5:
                    dwg.add(svgwrite.shapes.Line(start=(x0, y0), end=(x1, y1),stroke_width=rigidClusterStrokeWidth,stroke=colorForCluster,))
                
            
            
    dwg.saveas(fileName)
   
def forcePlotter(positions,radii,forces,systemSizeLx,systemSizeLz,forceScalar=1,outputFile=False,inputDwg=False,forceColor="red",threshold=5):
    
  


    
    if inputDwg != False:
        svgFile = inputDwg
    else:
        svgFile = svgwrite.Drawing(size=(systemSizeLx+systemSizeLx/10, systemSizeLz+systemSizeLz/10))
    
    
    #rescale magnitudes
    forceMagnitudes = forceScalar*forces[:,2:]
    #Get the particle ids of relevant particles
    contactingPairs = forces[:,:2]
    
        
    
    xPos = positions[:,0] + 11*systemSizeLx/20
    yPos = positions[:,1] + 11*systemSizeLz/20
    if inputDwg == False:
        #First plot all the circles
        #breakpoint()
        for i in range(0,len(radii)):
            svgFile.add(svgwrite.shapes.Circle(center=(xPos[i], yPos[i]), r=radii[i],fill='white',stroke='black',stroke_width=.1,))
            
    #Next plot all the frictional forces in red and the hydrodynamic forces in blue
    
    (numRows,_) =  np.shape(contactingPairs)
    
    for i in range(0,numRows):
        firstParticleIndex = contactingPairs[i,0]
        secondParticleIndex = contactingPairs[i,1] 

        xPos1 = xPos[int(firstParticleIndex)]
        yPos1 = yPos[int(firstParticleIndex)]
        
        xPos2 = xPos[int(secondParticleIndex)]
        yPos2 = yPos[int(secondParticleIndex)]
        
        forceMag = abs(forceMagnitudes[i][0])
        
        if ((xPos1-xPos2)**2 + (yPos1-yPos2)**2 < systemSizeLx/threshold):
            if forceMag != 0:
                svgFile.add(svgwrite.shapes.Line(start=(xPos1, yPos1), end=(xPos2, yPos2),stroke_width=forceMag,stroke=forceColor,))
                
    if outputFile==False:
        return svgFile
    else:
        svgFile.saveas(outputFile)  
        return svgFile
        
def returnMapPlotter(rigFileList,numParticles,shift=1,savePlot=False):
    """
    This function takes in a list of paths to different rig_ files and then makes the return maps plots for them.
    
 
        
        
    """
    
    height = 8
    width = 1.61803398875*height
    f, ax = matplotlib.pyplot.subplots(figsize=(width,height))
    
    ax.set_xlabel("$S_{max,\gamma}$",fontsize=25)
    ax.set_ylabel("$S_{max,\gamma+\delta}$, $\delta=$"+str(shift),fontsize=25)
    ax.set_ylim(0,numParticles)
    ax.set_xlim(0,numParticles)
    colorList = ["blue","orange","green","red","purple","brown","olive","cyan","black","lawngreen","indigo"]
    
    stressList = []
    for rigFile in rigFileList:
        
        result = re.search('_stress(.*)cl', rigFile)
        currentStress = result.group(1)
        stressList.append(float(currentStress))
        
    
        
    stressList = np.array(stressList)
    stressList = np.sort(stressList)
    
        
    
    
    
    for rigFile in rigFileList:
        
        result = re.search('_stress(.*)cl', rigFile)
        currentStress = result.group(1)
        
        colorCounter = np.where(stressList==float(currentStress))[0][0]
        
        clusterSizes, numBonds, clusterIDs = rigidClusterProcessor.rigFileReader(rigFile)
        largestClusters = np.zeros(len(clusterSizes))
        
        
        counter=0
        for currentClusterList in clusterSizes:
            largestClusters[counter] = np.max(currentClusterList)
            counter=counter+1

        

        #Now let's generate the return map with the correct shift.
        intialClusters = largestClusters[:-shift]
        finalClusters = largestClusters[shift:]

        
        #Now that we have the relevant data for the return map.  We just throw out the last element and throw out the first element then concatenate together
        returnMapData = np.vstack([intialClusters,finalClusters]).transpose()
    
        ax.plot(returnMapData[:,0],returnMapData[:,1],'o',color=colorList[colorCounter],label="$\sigma = $"+currentStress,alpha=0.5)
        
        
        
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    matplotlib.pyplot.subplots_adjust(bottom=0.18)
    matplotlib.pyplot.subplots_adjust(left=0.22)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.grid()
    
    if savePlot!=False:
        matplotlib.pyplot.savefig(savePlot)
    
    return ax
            
def viscosityVsRigidClusterPlotter(dataFileList,rigFileList,snapShotRange,maxClusterSize=False):
    
    #Let's pull out the needed rigid cluster data for each rig file
    
    medianDataHolder = np.zeros((len(rigFileList),2))
    
    counter = 0
    for rigFile in rigFileList:
        rigidClusterSizes,_ = rigidClusterProcessor.rigFileReader(rigFile,snapShotRange,False)
        
        result = re.search('_stress(.*)cl', rigFile)
        currentStress = float(result.group(1))
        
        holder = np.array([])
        for i in rigidClusterSizes:
            if maxClusterSize == False:
                holder = np.append(holder,i)
            else:
                holder = np.append(holder,np.max(i))
        holder = holder[holder != 0]
        medianClusterSize = np.median(holder)
        
        medianDataHolder[counter,0] = currentStress
        medianDataHolder[counter,1] = medianClusterSize
        counter=counter+1
        
    #Now let's get the viscosity
    
    viscosityDataHolder = np.zeros((len(dataFileList),2))
    
    counter = 0
    stressList = []
    for dataFile in dataFileList:
        
        
        result = re.search('_stress(.*)cl', dataFile)
        currentStress = float(result.group(1))
        stressList.append(currentStress)
        
        
        viscosityValues = rigidClusterProcessor.viscosityAverager(dataFile)
         
         
        viscosityDataHolder[counter,0] = currentStress
        viscosityDataHolder[counter,1] = np.mean(viscosityValues)
         
        counter=counter+1
    
    
    
    medianDataHolder = medianDataHolder[np.argsort(medianDataHolder[:, 0])]                                
    viscosityDataHolder = viscosityDataHolder[np.argsort(viscosityDataHolder[:, 0])]
    return stressList,viscosityDataHolder,medianDataHolder
                                              
def rigidLengthDistPlotter3D(rigFileList,parFileList,listOfStresses,numParticles,snapShotRange,Lx,nBins=500,rotation=False):
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    

    for rigFile, parFile, stress in zip(rigFileList, parFileList,listOfStresses):
        
        (xExtentHolder,_) = rigidClusterProcessor.rigidClusterLength(rigFile,parFile,numParticles,snapShotRange,rotation)
        
        allXExtents = np.array([])


        for i in range(0,len(xExtentHolder)):
            allXExtents = np.append(allXExtents,xExtentHolder[i])
    
        allXExtents = allXExtents/Lx
        
        hist, bin_edges = np.histogram(allXExtents,density=True,bins=nBins)
        bin_edge_average =np.zeros(len(bin_edges)-1)
        for i in range(0,len(bin_edges)-1):
            bin_edge_average[i] = (bin_edges[i+1]+bin_edges[i])/2
        
    
        
        X = bin_edge_average
        Z = hist
        
        
        ax.bar(X, Z, zs=stress, zdir='y',width = bin_edges[1]-bin_edges[0])
        
    return (fig,ax)

def rigidClusterSizeDist3D(rigFileList,snapShotRange,nBins,uniqueValues=True):
    
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    
    def log_tick_formatter(val, pos=None):
        return "{:.2e}".format(10**val)
    
    for rigFile in rigFileList:
        
        #Read in the rigid cluster sizes for each snapshot
        rigidClusterSizes,_ = rigidClusterProcessor.rigFileReader(rigFile,snapShotRange,False)
        
        
        #Use regular expressions to get the stress that the rigFile corresponds to
        result = re.search('_stress(.*)cl', rigFile)
        currentStress = float(result.group(1))
        
        
        #Let's put all of the arrays in rigidClusterSizes into one big array.
        allClusters = np.array([])
        for i in range(0,len(rigidClusterSizes)):
            
            if uniqueValues==True:
                allClusters = np.append(allClusters,np.unique(rigidClusterSizes[i]))
            else:
                allClusters = np.append(allClusters,rigidClusterSizes[i])
        
        #Now we copmute the histogram of the data.
        hist, bin_edges = np.histogram(allClusters,bins=nBins)
        
        #The histogram return variable bin_edges has one more entry than hist which will be problematic when plotting.
        #To fix this and also get the correct x-axis for "size of cluster" we will take the averages
        bin_edge_average =np.zeros(len(bin_edges)-1)
        for i in range(0,len(bin_edges)-1):
            bin_edge_average[i] = (bin_edges[i+1]+bin_edges[i])/2
        
        
        
        #Now we'll plot this particular stress in the 3D histogram
        X = bin_edge_average
        Z = hist/len(rigidClusterSizes)
        ax.bar(X, Z, zs=np.log10(currentStress), zdir='y',width = bin_edges[1]-bin_edges[0],label="$\sigma = $"+str(currentStress))
    
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.legend()
    
    return (fig,ax)

def rigidClusterSizeDist2D(rigFileList,snapShotRange,nBins,sMax=True):
    
    matplotlib.pyplot.clf()
    
    
    for rigFile in rigFileList:            
        #Read in the rigid cluster sizes for each snapshot
        rigidClusterSizes,_ = rigidClusterProcessor.rigFileReader(rigFile,snapShotRange,False)
        
        
        #Use regular expressions to get the stress that the rigFile corresponds to
        result = re.search('_stress(.*)cl', rigFile)
        currentStress = float(result.group(1))
        
        
        #Let's put all of the arrays in rigidClusterSizes into one big array.
        allClusters = np.array([])
        for i in range(0,len(rigidClusterSizes)):
            
            if sMax==True:
                allClusters = np.append(allClusters,np.max(rigidClusterSizes[i]))
            else:
                allClusters = np.append(allClusters,rigidClusterSizes[i])
            
        
        #Now we copmute the histogram of the data.
        kwargs = dict( alpha=0.3, bins=nBins,label="$\sigma = $"+str(currentStress),density=True)


        matplotlib.pyplot.hist(allClusters, **kwargs)
        
        
        
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel("Size of Cluster $S$")
    if sMax==True:
        matplotlib.pyplot.ylabel("$P(S_{max})$")
    else:
        matplotlib.pyplot.ylabel("$P(S)$")
    matplotlib.pyplot.xlim((2,2000))
    #matplotlib.pyplot.ylim((0,15))
    matplotlib.pyplot.yscale("log")
    matplotlib.pyplot.xscale("log")





    
    
    
    
    
    
    
    
        
        
        
        