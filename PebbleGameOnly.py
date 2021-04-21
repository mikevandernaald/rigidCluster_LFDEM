import matplotlib as mpl

mpl.use('Agg')

import random
import sys, os, glob

import Configuration as CF
import Pebbles as PB
import Hessian as HS
import Analysis as AN

import matplotlib.pyplot as plt

# Just change this directory below
foldername = os.getcwd() + '/' + sys.argv[1] + '/'
mu = 1

ThisConf = CF.Configuration(foldername, 'simulation', mu)
ThisConf.readSimdata(float(sys.argv[1]), True, 1)

ThisPebble = PB.Pebbles(ThisConf, 3, 3, 'nothing', False)
ThisPebble.play_game()

cidx, clusterall, clusterallBonds, clusteridx, BigCluster = ThisPebble.rigid_cluster(
)

ThisHessian = HS.Hessian(ThisConf)

ThisAnalysis = AN.Analysis(ThisConf, ThisPebble, ThisHessian, 0.01, False)
# zav,nm,pres,fxbal,fybal,torbal,mobin,mohist,sxx,syy,sxy,syx=ThisAnalysis.getStressStat()
# frac,fracmax,lenx,leny=ThisAnalysis.clusterStatistics()

# plt.figure(figsize=(10,7.5))

# fig1 = ThisAnalysis.plotStresses(True,False,False,True,False)
fig2 = ThisAnalysis.plotPebbles(True, True, False, True, False)

plt.savefig('./' + sys.argv[1] + '/' + sys.argv[1] + '.png')
