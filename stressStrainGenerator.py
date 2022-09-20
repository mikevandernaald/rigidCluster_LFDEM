# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 12:53:20 2022

@author: mikev
"""

import rigidClusterProcessor
import numpy as np
import rigidClusterPlotter
import os 
import itertools




def extractGapsAndTangentialSpringStretch(intFile,k_t,snapShotRange=False):
    """
    This function takes in an intFile and gets out the interparticle stresses from the
    tangential contact forces and the normal contact forces. 
    """
    
    #If the optional variable snapShotRange is not set then process all snapshots.  Otherwise set the correct range.

    
    with open(intFile) as f1:
        fileLines = f1.readlines()[24:]
        
    #We'll first find every line in intFile that starts with "#" as that is a line where a new snapshop starts.
    counter=0
    linesWhereDataStarts=np.array([])
    for lines in fileLines:
        if (np.shape(np.fromstring(lines,sep=' '))[0]==0) & ("#" in str(lines)):
            linesWhereDataStarts = np.append(linesWhereDataStarts, counter)
        counter=counter+1
    
    #Calculate the total number of snapshots
    numSnapshots = np.shape(linesWhereDataStarts)[0]
    #If the optional variable snapShotRange is not set then process all snapshots.  Otherwise set the correct range.
    if snapShotRange == False:
        lowerSnapShotRange = 0
        upperSnapShotRange = numSnapshots
    else:
        lowerSnapShotRange = snapShotRange[0]
        upperSnapShotRange = snapShotRange[1]
        
        
    #Now lets make a python list to store all the different contacts for each snapshot
    contactInfo = [0] * (upperSnapShotRange-lowerSnapShotRange)

    #Now we'll loop through each snapshot and store only the first three columns.  This should hopefully make this less expensive.
    #The first column is the first particle index, the second is the second particle index and the final column tells us the contact type.
    #We will also be ignoring any interaction where the contact type is 0 as that is a hydrodynamic interaction.  
    #After we identify relevant contacts that give frictional stresses
    counter = 0
    for i in range(lowerSnapShotRange,upperSnapShotRange):
        if i==numSnapshots-1:
            currentContacts = np.genfromtxt(itertools.islice(fileLines, int(linesWhereDataStarts[i]), int(linesWhereDataStarts[i + 1])),usecols=(0,1,2,3,4,5,6,12,13,14))
            #Only consider contacts that are frictionally engaged
            currentContacts = currentContacts[np.where(currentContacts[:, 2] > 1), :][0]
            #We need to extract the tangential spring stretches for each contact
            springStretches = np.sqrt(currentContacts[:,7]**2+currentContacts[:,8]**2+currentContacts[:,9]**2)/k_t
            if len(currentContacts)==0:
                contactInfo[counter] = np.array([[0],[0]])
            else:    
                contactInfo[counter] =  np.concatenate((currentContacts[:,:7],np.expand_dims(springStretches, axis=1)),axis=1)
            
        else:
            currentContacts = np.genfromtxt(itertools.islice(fileLines, int(linesWhereDataStarts[i]), int(linesWhereDataStarts[i + 1])),usecols=(0,1,2,3,4,5,6,12,13,14))
            #Only consider contacts that are frictionally engaged
            currentContacts = currentContacts[np.where(currentContacts[:, 2] > 1), :][0]
            #We need to extract the tangential spring stretches for each contact
            springStretches = np.sqrt(currentContacts[:,7]**2+currentContacts[:,8]**2+currentContacts[:,9]**2)/k_t
            if len(currentContacts)==0:
                contactInfo[counter] = np.array([[0],[0]])
            else:    
                contactInfo[counter] =  np.concatenate((currentContacts[:,:7],np.expand_dims(springStretches, axis=1)),axis=1)

        del currentContacts
        counter=counter+1

    #We no longer need fileLines and it takes up a lot of RAM so we can delete it (not sure if this is needed, python interpreters are pretty good about this stuff)
    del fileLines
    return contactInfo
    
    
    