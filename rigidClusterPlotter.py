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

from PIL import Image



def dataExtractorLFDEMFrictionalForces(parFile,intFile,snapShotRange=False,intFileColumns=False):
    
    
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
    for i in range(lowerSnapShotRange,upperSnapShotRange):
        if i==numSnapshots-1:
            #If there is a 0 in the third column then that means the particles are not in contact and we can throw that row our.
            currentContacts = np.genfromtxt(itertools.islice(fileLines, int(linesWhereDataStarts[i]), int(numLines)),usecols=(0, 1, 2,3,5,12,14))
            currentContacts = currentContacts[np.where(currentContacts[:, 2] > 1), :][0]
            xComponentsOfFrictionalForces = currentContacts[:,3]+currentContacts[:,5]
            zComponentsOfFrictionalForces = currentContacts[:,4]+currentContacts[:,6]
            magnitudes = np.sqrt(xComponentsOfFrictionalForces**2+zComponentsOfFrictionalForces**2)
            if len(currentContacts)==0:
                contactInfo[counter] = 0
            else:    
                contactInfo[counter] = np.concatenate((np.expand_dims(currentContacts[:,0], axis=1),np.expand_dims(currentContacts[:,1], axis=1),np.expand_dims(magnitudes, axis=1)),axis=1)
            
        else:
            currentContacts = np.genfromtxt(itertools.islice(fileLines, int(linesWhereDataStarts[i]), int(linesWhereDataStarts[i + 1])),usecols=(0, 1, 2,3,5,12,14))
            currentContacts = currentContacts[np.where(currentContacts[:, 2] > 1), :][0]
            xComponentsOfFrictionalForces = currentContacts[:,3]+currentContacts[:,5]
            zComponentsOfFrictionalForces = currentContacts[:,4]+currentContacts[:,6]
            magnitudes = np.sqrt(xComponentsOfFrictionalForces**2+zComponentsOfFrictionalForces**2)
            if len(currentContacts)==0:
                contactInfo[counter] = 0
            else:    
                contactInfo[counter] = np.concatenate((np.expand_dims(currentContacts[:,0], axis=1),np.expand_dims(currentContacts[:,1], axis=1),np.expand_dims(magnitudes, axis=1)),axis=1)
        del currentContacts
        counter=counter+1

    #We no longer need fileLines and it takes up a lot of RAM so we can delete it (not sure if this is needed, python interpreters are pretty good about this stuff)
    del fileLines
    
    return (contactInfo,positionData,particleRadii,systemSizeLx,systemSizeLz,numParticles)



def dataExtractorLFDEMBothForces(parFile,intFile,snapShotRange=False):
    
    
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
    contactInfoFrictional = [0] * (upperSnapShotRange-lowerSnapShotRange)
    contactInfoHydro = [0] * (upperSnapShotRange-lowerSnapShotRange)

    #Now we'll loop through each snapshot and store only the first three columns.  This should hopefully make this less expensive.
    #The first column is the first particle index, the second is the second particle index and the final column tells us the contact type.
    #We will also be ignoring any interaction where the contact type is 0 as that is a hydrodynamic interaction.
    counter = 0
    for i in range(lowerSnapShotRange,upperSnapShotRange):
        if i==numSnapshots-1:
            currentContacts = np.genfromtxt(itertools.islice(fileLines, int(linesWhereDataStarts[i]), int(numLines)),usecols=(0,1,2,3,5,7,8,10,11,12,14))
        else:
            currentContacts = np.genfromtxt(itertools.islice(fileLines, int(linesWhereDataStarts[i]), int(linesWhereDataStarts[i + 1])),usecols=(0,1,2,3,5,7,8,10,11,12,14))
            
        magnitudeOfHydroForces = np.sqrt(currentContacts[:,5]**2+currentContacts[:,6]**2+currentContacts[:,7]**2)
        magnitudeOfContactForces = np.sqrt(currentContacts[:,8]**2+currentContacts[:,9]**2+currentContacts[:,10]**2)
            
        if len(currentContacts)==0:
            contactInfoFrictional[counter] = 0
            contactInfoHydro[counter] = 0
        else:
            contactInfoFrictional[counter] = np.concatenate((np.expand_dims(currentContacts[:,0], axis=1),np.expand_dims(currentContacts[:,1], axis=1),np.expand_dims(magnitudeOfContactForces, axis=1)),axis=1)
            contactInfoHydro[counter] = np.concatenate((np.expand_dims(currentContacts[:,0], axis=1),np.expand_dims(currentContacts[:,1], axis=1),np.expand_dims(magnitudeOfHydroForces, axis=1)),axis=1)
        del currentContacts
        counter=counter+1
    
    return (contactInfoHydro,contactInfoFrictional,positionData,particleRadii,systemSizeLx,systemSizeLz,numParticles)

 
    
    
def rigidClusterPlotter(fileName,currentPosData,particleRadii,systemSizeLx,systemSizeLz,confObj,pebblesObj,analysisObj,rigidClusterStrokeWidth):

    
    dwg = svgwrite.Drawing(fileName, size=(systemSizeLx+systemSizeLx/10, systemSizeLz+systemSizeLz/10))
    
    xPos = currentPosData[:,0]
    yPos = currentPosData[:,1]
    #First plot all the circles
    numParticles = len(particleRadii)
    for i in range(0,numParticles):
        dwg.add(svgwrite.shapes.Circle(center=(xPos[i], yPos[i]), r=particleRadii[i],fill='white',stroke='black',stroke_width=.1,))

    numUniqueColorsNeeded = np.shape(np.unique(pebblesObj.cluster))[0]-1
    

    colorForCluster = matplotlib.colors.to_hex([0.085, 0.532, 0.201])


    if numUniqueColorsNeeded!=0: 
        for k in range(len(pebblesObj.Ifull)):
            # this version depends on pebble numbering, so use 2nd version
            y0, y1, x0, x1 = confObj.getConPos2(pebblesObj.Ifull[k],pebblesObj.Jfull[k])
            x0 = x0 + 11 * systemSizeLx / 20
            x1 = x1 + 11 * systemSizeLx / 20
            y0 = y0 + 11 * systemSizeLz / 20
            y1 = y1 + 11 * systemSizeLz / 20
            if (x0-x1)**2 + (y0-y1)**2 < systemSizeLx/5:
                if (pebblesObj.cluster[k] != -1):
                    if (k >= analysisObj.ncon):
                        dwg.add(svgwrite.shapes.Line(start=(x0, y0), end=(x1, y1),stroke_width=rigidClusterStrokeWidth,stroke=colorForCluster,))
                    else:
                        dwg.add(svgwrite.shapes.Line(start=(x0, y0), end=(x1, y1),stroke_width=rigidClusterStrokeWidth,stroke=colorForCluster,))
                    
    dwg.save()            
                
    

    
def generatePlots(topDir,parFile,intFile,snapShot,scalarFrictionalForces,scalarHydroForces,rigidClusterStrokeWidth,name):
    
    
    topDirRigidCluster = os.path.join(topDir,"rigidClusters")
    topDirHydro = os.path.join(topDir,"hydroForces")
    topDirFrictional = os.path.join(topDir,"frictionalForces")
    
    
    fileNameRigid = os.path.join(topDirRigidCluster,name+"rigidCluster.svg")
    #fileNameForces= os.path.join(topDir,name+"forces.svg")
    
    (contactInfoHydro,contactInfoFrictional,positionData,particleRadii,systemSizeLx,systemSizeLz,numParticles) = dataExtractorLFDEMBothForces(parFile,intFile,[snapShot,snapShot+1])


    (clusterHolder,ThisPebble,ThisConf) = rigidClusterProcessor.pebbleGame_LFDEMSnapshot(parFile,intFile,False,[snapShot,snapShot+1])
    ThisHessian = HS.Hessian(ThisConf)

    ThisAnalysis = AN.Analysis(ThisConf, ThisPebble, ThisHessian, 0.01, False)
    
    
    
    frictionalAndHydroForcePlotter(topDirHydro,topDirFrictional,name,particleRadii,positionData[:,:,0],contactInfoHydro[0][:,:2],contactInfoFrictional[0][:,2],contactInfoHydro[0][:,2],systemSizeLx,systemSizeLz,5,scalarFrictionalForces,scalarHydroForces)
    

    rigidClusterPlotter(fileNameRigid,positionData[:,:,0],particleRadii,systemSizeLx,systemSizeLz,ThisConf,ThisPebble,ThisAnalysis,rigidClusterStrokeWidth)


def rigidCluserMovieMaker(topDir,parFile,intFile,scalarFrictionalForces,scalarHydroForces,rigidClusterStrokeWidth,snapShotRange=False):
    os.mkdir(os.path.join(topDir,"rigidClusters"))
    os.mkdir(os.path.join(topDir,"hydroForces"))
    os.mkdir(os.path.join(topDir,"frictionalForces"))
    
    topDirRigidCluster = os.path.join(topDir,"rigidClusters")
    topDirHydro = os.path.join(topDir,"hydroForces")
    topDirFrictional = os.path.join(topDir,"frictionalForces")
    
    (contactInfoHydro,contactInfoFrictional,positionData,particleRadii,systemSizeLx,systemSizeLz,numParticles) = dataExtractorLFDEMBothForces(parFile,intFile,snapShotRange)
    clusterHolder= rigidClusterProcessor.pebbleGame_LFDEMSnapshot(parFile,intFile,False,snapShotRange)

    print("Done computing rigid cluster information.  Starting movie making..")
    if snapShotRange == False:
        snapShots = np.linspace(0, len(clusterHolder), len(clusterHolder) + 1)
    else:
        snapShots = np.linspace(snapShotRange[0], snapShotRange[1], snapShotRange[1] - snapShotRange[0])

    counter=0
    for i in snapShots:
        fileName = os.path.join(topDir,str(i))
        
        fileNameRigid = os.path.join(topDirRigidCluster,str(i)+"_rigidCluster.svg")

        ThisPebble = clusterHolder[int(counter)][5]
        ThisConf = clusterHolder[int(counter)][6]
        ThisHessian = HS.Hessian(ThisConf)
        ThisAnalysis = AN.Analysis(ThisConf, ThisPebble, ThisHessian, 0.01, False)
        
        positionData[:,1,int(counter)] = positionData[:,1,int(counter)] + 11*systemSizeLx/20
        positionData[:,0,int(counter)] = positionData[:,0,int(counter)] + 11*systemSizeLx/20
        
        frictionalAndHydroForcePlotter(topDirHydro,topDirFrictional,fileName,particleRadii,positionData[:,:,int(counter)],contactInfoHydro[int(counter)][:,:2],contactInfoFrictional[int(counter)][:,2],contactInfoHydro[int(counter)][:,2],systemSizeLx,systemSizeLz,5,scalarFrictionalForces,scalarHydroForces)
        rigidClusterPlotter(fileNameRigid,positionData[:,:,int(counter)],particleRadii,systemSizeLx,systemSizeLz,ThisConf,ThisPebble,ThisAnalysis,rigidClusterStrokeWidth)
        counter=counter+1
        
        
def rigidClusterMovieComposer(hydroDir,frictionalDir,rigidClusterDir,outputDir):
    
    
    hydroFiles = os.listdir(hydroDir)
    frictionalFiles = os.listdir(frictionalDir)
    rigidClusterFiles = os.listdir(rigidClusterDir)
    
    
    
    for i in range(0,len(frictionalFiles)):
        
        currentHydroFile = os.path.join(hydroDir,hydroFiles[i])
        currentFricitonalFile = os.path.join(frictionalDir,frictionalFiles[i])
        currentRigidClusterFile = os.path.join(rigidClusterDir,rigidClusterFiles[i])
        
        images = [Image.open(x) for x in [currentHydroFile, currentFricitonalFile, currentRigidClusterFile]]
        widths, heights = zip(*(i.size for i in images))
        
        total_width = sum(widths)
        max_height = max(heights)
        
        new_im = Image.new('RGB', (total_width, max_height))
        

        
        x_offset = 0
        for im in images:
          new_im.paste(im, (x_offset,0))
          x_offset += im.size[0]
        
        
        
        rgba = np.array(new_im)

        # Make image transparent white anywhere it is transparent
        rgba[rgba[...,-1]==0] = [255,255,255]
        
        
        Image.fromarray(rgba).save(os.path.join(outputDir,str(i)+".png"))



def forcePlotter(positions,radii,forces,systemSizeLx,systemSizeLz,forceScalar=1,outputFile=False,inputDwg=False,forceColor="red",threshold=5):
    
    if outputFile==False:
        svgFile = svgwrite.Drawing(size=(systemSizeLx+systemSizeLx/10, systemSizeLz+systemSizeLz/10))
    else: 
        svgFile = svgwrite.Drawing(outputFile, size=(systemSizeLx+systemSizeLx/10, systemSizeLz+systemSizeLz/10))
    
    if inputDwg != False:
        svgFile = inputDwg
    
    
    #rescale magnitudes
    forceMagnitudes = forceScalar*forces[:,2:]
    #Get the particle ids of relevant particles
    contactingPairs = forces[:,:2]
    

    if inputDwg == False:
        xPos = positions[:,0]
        yPos = positions[:,1]
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
        
        forceMag = forceMagnitudes[i]
        
        if ((xPos1-xPos2)**2 + (yPos1-yPos2)**2 < systemSizeLx/threshold):
            if forceMag != 0:
                svgFile.add(svgwrite.shapes.Line(start=(xPos1, yPos1), end=(xPos2, yPos2),stroke_width=forceMag,stroke=forceColor,))
                
        if outputFile==False:
            return svgFile
        else:
            svgFile.save()  
            return svgFile
        
        


def returnMapPlotter(rigFileList,numParticles):
    """
    This function takes in a list of paths to different rig_ files and then makes the return maps plots for them.
    
 
        
        
    """
    
    height = 8
    width = 1.61803398875*height
    f, ax = matplotlib.pyplot.subplots(figsize=(width,height))
    
    ax.set_xlabel("$S_{max,\gamma}$",fontsize=25)
    ax.set_ylabel("$S_{max,\gamma+1}$",fontsize=25)
    ax.set_ylim(0,numParticles)
    ax.set_xlim(0,numParticles)
    colorList = ["blue","orange","green","red","purple","brown","pink","gray","olive","cyan","black"]
    
    bigCounter=0
    for rigFile in rigFileList:
        
        result = re.search('_stress(.*)cl', rigFile)
        currentStress = result.group(1)
        
        clusterSizes, numBonds, clusterIDs = rigidClusterProcessor.rigFileReader(rigFile)
        largestClusters = np.zeros(len(clusterSizes))
        
        
        counter=0
        for currentClusterList in clusterSizes:
            largestClusters[counter] = np.max(currentClusterList)
            counter=counter+1


        #Now that we have the relevant data for the return map.  We just throw out the last element and throw out the first element then concatenate together
        returnMapData = np.vstack([largestClusters[:-1],largestClusters[1:]]).transpose()
    
        ax.plot(returnMapData[:,0],returnMapData[:,1],'o',color=colorList[bigCounter],label="$\sigma = $"+currentStress,alpha=0.5)
        
        bigCounter = bigCounter+1
        
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    matplotlib.pyplot.subplots_adjust(bottom=0.18)
    matplotlib.pyplot.subplots_adjust(left=0.22)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.grid()
    
    return ax
            

    
    
    
    
        
        
        
        