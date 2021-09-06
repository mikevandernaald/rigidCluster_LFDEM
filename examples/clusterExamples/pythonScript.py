# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 09:46:19 2021

@author: Mike van der Naald
"""

import sys
#Put the path to the rigidCluster_LFDEM github repo here.  Otherwise the cluster doesn't know where our libraries live.
sys.path.insert(0, r"/home/mrv/pebbleCode/rigidCluster_LFDEM")
import rigidClusterProcessor


rigidClusterProcessor.rigFileGenerator(r"/project2/jaeger/abhinendra/rigid_cluster_data/N_2000/mu_1/VF0.77",r"/project2/jaeger/abhinendra/rigid_cluster_data/N_2000/mu_1/VF0.77")
