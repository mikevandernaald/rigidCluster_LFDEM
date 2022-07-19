# -*- coding: utf-8 -*-
"""
Example file for the github repo
"""

import rigidClusterProcessor
import numpy as np
from matplotlib import pyplot



#Generate rig_ files

#topDir points to the directory with the par_ and int_ files which we wish process into rig_ files
topDir = r"D:\data\rigidClusterData\dataFromCluster\mu_1\VF0.78"
#output the rig_ files to the same place with the par_ and int_ files
outputDir = topDir
#Process snapshots starting with the 50th snapshot going all the way to the last snapshot
snapShotRange = [50,-1]
#Set it so that the rig_ files contain the particle IDs for each cluster
reportIDS = True
#Read in stress controlled data
stressControlled=True

rigidClusterProcessor.rigFileGenerator(topDir,outputDir,snapShotRange,reportIDS,stressControlled)




#Read in rig_ files
#Path for rig_ file
rigFile = r"D:\data\rigidClusterData\dataFromCluster\mu_1\VF0.78\rig_D2N2000VF0.78Bidi1.4_0.5Square_1_pardata_phi0.78_stress50cl.dat"
#Read in snapshots starting with the 50th snapshot going all the way to the last snapshot
snapShotRange = [50,-1]
#Read in particle IDs
reportIDS = True
(rigidClusterSizes,numBonds,clusterIDs) = rigidClusterProcessor.rigFileReader(rigFile,snapShotRange,reportIDS)


#Plot a histogram of all particle sizes.
particleSizeHolder = np.array([])

for clusterLists in rigidClusterSizes:
    particleSizeHolder = np.append(particleSizeHolder,np.array(clusterLists))
    
    
pyplot.hist(particleSizeHolder)
pyplot.xlabel(r"Cluster Size S")
pyplot.ylabel("Counts")