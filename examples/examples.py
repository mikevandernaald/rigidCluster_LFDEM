# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 11:51:05 2021

@author: mikev
"""

import rigidClusterProcessor
import os
import rigidClusterPlotter


"""
Example of how to use the rigidClusterProcessor
"""
#Change this to where the Github examples folder is
dataDir = r"C:\Users\mikev\Documents\GitHub\rigidCluster_LFDEM\examples"


#These are just example LF_DEM files for a system with phi=0.78, stress = 20.
intFile = os.path.join(dataDir,r"int_D2N2000VF0.78Bidi1.4_0.5Square_1_pardata_phi0.78_stress20cl.dat")
parFile = os.path.join(dataDir,r"par_D2N2000VF0.78Bidi1.4_0.5Square_1_pardata_phi0.78_stress20cl.dat")

#This is the snapshot range that we want the cluster information for.
snapShotRange = [130,132]


#Each element in the list clusterInformation contains the cluster info for a particle snapshot.
clusterInformation = rigidClusterProcessor.pebbleGame_LFDEMSnapshot( parFile,intFile,snapShotRange )


clusterInfo_snapShot130 = clusterInformation[0]

clusters=''

for i in clusterInfo_snapShot130[0]:
    clusters = clusters+str(i)+", "
print("Snapshot 130 has clusters of the size:  " + clusters)




"""
Examples of how to use the rigidClusterPlotter
"""


imageOutputDir = r"C:\Users\mikev\Documents\GitHub\rigidCluster_LFDEM\examples"

#Change this to where the Github examples folder is
dataDir = r"C:\Users\mikev\Documents\GitHub\rigidCluster_LFDEM\examples"


#These are just example LF_DEM files for a system with phi=0.78, stress = 20.
intFile = os.path.join(dataDir,r"int_D2N2000VF0.78Bidi1.4_0.5Square_1_pardata_phi0.78_stress20cl.dat")
parFile = os.path.join(dataDir,r"par_D2N2000VF0.78Bidi1.4_0.5Square_1_pardata_phi0.78_stress20cl.dat")


snapShot = 130


scalarFrictionalForces = 1/20
scalarHydroForces = 1/20
rigidClusterStrokeWidth = .5
name = "exampleOutput"


rigidClusterPlotter.generatePlots(imageOutputDir,parFile,intFile,snapShot,scalarFrictionalForces,scalarHydroForces,rigidClusterStrokeWidth,name)