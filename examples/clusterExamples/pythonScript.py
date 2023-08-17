# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 09:46:19 2021

@author: Mike van der Naald
"""

import sys
#Put the path to the rigidCluster_LFDEM github repo here.  Otherwise the cluster doesn't know where the library is.
sys.path.insert(0, r"/home/mrv/pebbleCode/rigidCluster_LFDEM")
import rigidClusterProcessor

#topDir is where the collection of par_ and int_ files live that you want to generate corresponding rig_ files for.
#outputDir is where the rig_ files will be output to
topDir = r"/project2/jaeger/abhinendra/rigid_cluster_data/N_2000/mu_1/VF0.77"
outputDir = topDir
rigidClusterProcessor.rigFileGenerator(topDir,outputDir)
