# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 11:51:05 2021

@author: mikev
"""

import rigidClusterProcessor
import os
import numpy as np


"""
Generating a rig_ file
rig_ files have information about how many clusters there are, their size in both connections and particles, as well as the particle ID's of the
participating clusters.
"""

#Path to the directory that has par_ and int_ files that you want to make corresponding rig_ files for
topDir = r"C:\Users\mikev\Documents\temp"
#Path where rig_ files are outputted
outputDir = topDir
#snapShotRange gives the range of snapshots to process.  We set it to False if we want it to process all snapshots
#Alternatively one could set snapShotRange = [0,5] to process the first five snapshots.
snapShotRange = False
#reportIDs tells the function to report not only the size of each cluster but also which particles are participating
#in each cluster.  Setting it to True tells it to report the particle IDs of the particles in a cluster into the rig file
reportIDS = True

rigidClusterProcessor.rigFileGenerator(topDir,outputDir,snapShotRange,reportIDS)


"""
Reading rig_ files
"""

#rigFile is the path to the rig file you want to read in.
rigFile = r"C:\Users\mikev\Documents\temp\rig_D2N2000VF0.8Bidi1.4_0.5Square_1_pardata_phi0.8_stress10cl.dat"
#snapShotRange gives the range of snapshots to process.  We set it to False if we want it to process all snapshots
#Alternatively one could set snapShotRange = [0,5] to process the first five snapshots.
snapShotRange = False
#reportIDs tells the function read in IDs
readInIDS = True


clusterSizes, numBonds, clusterIDs = rigidClusterProcessor.rigFileReader(rigFile,snapShotRange,readInIDS)

#clusterSizes is a list.  Each element of the list is a numpy array that contains
#integers with sizes of



