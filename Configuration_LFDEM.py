# Silke Henkes, 29.08.18:
# Created Configuration class to read both experimental and simulation data
# Detangled code parts, contains:
# - Read-in for all circumstances
# - Analysis functions which depend on particle positions only, called by Analysis and others

# Silke Henkes 11.07.2013: Accelerated version of pebble game and rigid cluster code
#!/usr/bin/python

"""
This code was modified by Mike van der Naald on 2/11/2021.
The configuration class was modified to not read in text files for the data but instead have them be passed in
as arguments to the class constructor.  This is nice for feeding in dense suspension configurations from LF_DEM.
"""


import sys, os, glob
import numpy as np
sys.setrecursionlimit(1500000)


class Configuration:

    #=================== Create a new configruation ===============
    # Current choices: either simulation or experimental through datatype. Parameters are either read (simulation) or passed to the configuration via optional arguments (experiment)
    def __init__(self,numParticles,
                 systemSize,
                 strainRate,
                 radii
                 ):
        self.addBoundary = False
        self.getParameters(numParticles,systemSize,radii)
        # Simulation data has periodic boundary conditions and also angles as output
        self.periodic = True
        self.hasAngles = True
        # basis for mass computations
        self.density = 1.0
        # Prefactor of the stiffness coefficients
        self.stiffness = 1.0
        # radius conversion factor
        self.rconversion = 1.0
        self.height = 1.0
        self.width = 1.0


    #======= Simulation parameter read-in =====================
    # use Parameter read-in as function called by the constructor
    def getParameters(self,numParticles,systemSize,radii):
        self.N = numParticles
        self.L = systemSize
        self.gammadot = 0
        self.Lx = self.L
        self.Ly = self.L
        self.rad = radii

        #These parameters are not used in the RigidCluster calculation.  As far as I can tell, they're only used when
        #you don't give the contact network to the code and the code has to calculate it.  Since LF_DEM returns this data
        #we don't need to have the code calcualte it.
        self.krep = 1
        self.mu = 1
        self.xi = 1
        self.phi = .5

    #######========== Simulation data read-in ==================
    # snap is the label, and distSnapshot tells me how many time units they are apart (in actual time, not steps)
    def readSimdata(self, positionData,contactData,snap, distSnapshot0=100.0):
        self.distSnapshot = distSnapshot0
        self.strain = self.gammadot * (1.0 * snap) * self.distSnapshot

        self.x = positionData[:, 1]
        self.y = positionData[:, 0]
        #These other variables are not needed but we'll set them to zero now
        self.alpha = np.zeros(self.N)
        self.dx = np.zeros(self.N)
        self.dy = np.zeros(self.N)
        self.dalpha = np.zeros(self.N)
        del positionData



        self.I = list(contactData[:, 0].astype(int))
        self.J = list(contactData[:, 1].astype(int))
        self.fullmobi = contactData[:, 2].astype(int)-2
        #These other variables are not needed but we'll set them to zero now
        self.nx = np.zeros(np.shape(contactData)[0])
        self.ny = np.zeros(np.shape(contactData)[0])
        self.fnor = np.zeros(np.shape(contactData)[0])
        self.ftan = np.zeros(np.shape(contactData)[0])
        del contactData

        self.x -= self.L * np.round(self.x / self.L)
        self.y -= self.L * np.round(self.y / self.L)
        self.ncon = len(self.I)

    #### ======================== Boundary integration =======================================================
    def AddBoundaryContacts(self, threshold=20, Brad=20.0):
        self.addBoundary = True
        # Threshold to check if a particle is close enough to walls.
        upidx = np.argmax(self.y)
        downidx = np.argmin(self.y)
        leftidx = np.argmin(self.x)
        rightidx = np.argmax(self.x)

        # Boundary posiitons:
        # coordinates of virtual boundary particles: in the middle, one Brad off from the edge of the outermost particle
        up = self.y[upidx]
        yup = up + self.rad[upidx]
        down = self.y[downidx]
        ydown = down - self.rad[downidx]
        left = self.x[leftidx]
        xleft = left - self.rad[leftidx]
        right = self.x[rightidx]
        xright = right + self.rad[rightidx]

        # coordinates of virtual boundary particles: in the middle, one Brad off from the edge of the outermost particle
        Boundaries = np.zeros(
            (4, 3))  # Four boundary particles with their x,y and rad
        Boundaries[0, :] = [(left + right) * 0.5, yup + Brad, Brad]
        Boundaries[1, :] = [(left + right) * 0.5, ydown - Brad, Brad]
        Boundaries[2, :] = [xleft - Brad, (up + down) * 0.5, Brad]
        Boundaries[3, :] = [xright + Brad, (up + down) * 0.5, Brad]

        # Find the particles in contact with the boundary, and label correctly
        self.bindices = [self.N, self.N + 1, self.N + 2, self.N + 3]
        padd = []
        labels = []
        pup = np.nonzero(np.abs(self.y + self.rad - yup) < threshold)[0]
        padd.extend(pup)
        labels.extend([0 for k in range(len(pup))])
        pdown = np.nonzero(np.abs(self.y - self.rad - ydown) < threshold)[0]
        padd.extend(pdown)
        labels.extend([1 for k in range(len(pdown))])
        pleft = np.nonzero(np.abs(self.x - self.rad - xleft) < threshold)[0]
        padd.extend(pleft)
        labels.extend([2 for k in range(len(pleft))])
        pright = np.nonzero(np.abs(self.x + self.rad - xright) < threshold)[0]
        padd.extend(pright)
        labels.extend([3 for k in range(len(pright))])

        fullmobi_add = []
        fnor_add = []
        ftan_add = []
        nx_add = []
        ny_add = []
        for k in range(len(padd)):
            # does this guy have neighbours?
            neii = np.nonzero(self.I[:self.ncon] == padd[k])[0]
            neij = np.nonzero(self.J[:self.ncon] == padd[k])[0]
            # if yes add the boundary contacts
            if (len(neii) > 0 or len(neij) > 0):
                self.I.append(self.bindices[labels[k]])
                self.J.append(padd[k])
                if (labels[k]) == 0:
                    nx0 = 0
                    ny0 = -1
                elif (labels[k] == 1):
                    nx0 = 0
                    ny0 = 1
                elif (labels[k] == 2):
                    nx0 = 1
                    ny0 = 0
                else:
                    nx0 = -1
                    ny0 = 0
                # compute force on this contact by force balance
                # two minus signs on second part cancel out
                ftotx = np.sum(self.fnor[neii] * self.nx[neii] -
                               self.ftan[neii] * self.ny[neii]) - np.sum(
                                   self.fnor[neij] * self.nx[neij] -
                                   self.ftan[neij] * self.ny[neij])
                ftoty = np.sum(self.fnor[neii] * self.ny[neii] +
                               self.ftan[neii] * self.nx[neii]) - np.sum(
                                   self.fnor[neij] * self.ny[neij] +
                                   self.ftan[neij] * self.nx[neij])
                # (fx*nx+fy*ny)
                fnor0 = ftotx * nx0 + ftoty * ny0
                # (fx*(-ny)+fy*nx)
                ftan0 = ftotx * (-ny0) + ftoty * nx0
                #print ftan0
                if (abs(ftan0) / fnor0 > self.mu):
                    fullmobi_add.append(1)
                else:
                    fullmobi_add.append(0)
                fnor_add.append(fnor0)
                ftan_add.append(ftan0)
                nx_add.append(nx0)
                ny_add.append(ny0)
        # Finally stick it at the end of the existing data
        self.x = np.concatenate((self.x, Boundaries[:, 0]))
        self.y = np.concatenate((self.y, Boundaries[:, 1]))
        self.rad = np.concatenate((self.rad, Boundaries[:, 2]))
        self.fnor = np.concatenate((self.fnor, np.array(fnor_add)))
        self.ftan = np.concatenate((self.ftan, np.array(ftan_add)))
        self.fullmobi = np.concatenate((self.fullmobi, np.array(fullmobi_add)))
        self.nx = np.concatenate((self.nx, np.array(nx_add)))
        self.ny = np.concatenate((self.ny, np.array(ny_add)))
        self.ncon = len(self.I)
        self.N += 4
        print("Added boundaries!")

    def AddNextBoundaryContacts(self, threshold=15, Brad=20.0):
        # Threshold to check if a particle is close enough to walls.
        upidx = np.argmax(self.ynext)
        downidx = np.argmin(self.ynext)
        leftidx = np.argmin(self.xnext)
        rightidx = np.argmax(self.xnext)

        # Boundary posiitons:
        # coordinates of virtual boundary particles: in the middle, one Brad off from the edge of the outermost particle
        up = self.ynext[upidx]
        yup = up + Brad + self.radnext[upidx]
        down = self.ynext[downidx]
        ydown = down - Brad - self.radnext[downidx]
        left = self.xnext[leftidx]
        xleft = left - Brad - self.radnext[leftidx]
        right = self.xnext[rightidx]
        xright = right + Brad + self.radnext[rightidx]

        # coordinates of virtual boundary particles: in the middle, one Brad off from the edge of the outermost particle
        Boundaries = np.zeros(
            (4, 3))  # Four boundary particles with their x,y and rad
        Boundaries[0, :] = [(left + right) * 0.5, yup, Brad]
        Boundaries[1, :] = [(left + right) * 0.5, ydown, Brad]
        Boundaries[2, :] = [xleft, (up + down) * 0.5, Brad]
        Boundaries[3, :] = [xright, (up + down) * 0.5, Brad]

        self.xnext = np.concatenate((self.xnext, Boundaries[:, 0]))
        self.ynext = np.concatenate((self.ynext, Boundaries[:, 1]))
        self.radnext = np.concatenate((self.radnext, Boundaries[:, 2]))

        self.dx = self.xnext - self.x
        self.dy = self.ynext - self.y
        self.Nnext += 4

    #### ======================== Analysis helper functions	 =================================================

    # computes basic contact, force, torque, and stress statistics
    def getStressStat(self):
        fsumx = np.empty((self.N, ))
        fsumy = np.empty((self.N, ))
        torsum = np.empty((self.N, ))
        #------- those useful for plotting ----
        self.prepart = np.empty((self.N, ))
        self.sxxpart = np.empty((self.N, ))
        self.syypart = np.empty((self.N, ))
        self.sxypart = np.empty((self.N, ))
        self.syxpart = np.empty((self.N, ))
        #-------------------------------------
        for u in range(self.N):
            # problem - self.I doesn't seem to behave as an integer ...
            # apparently this only works with arrays, not with lists??
            c1 = np.nonzero(np.array(self.I) == u)
            c2 = np.nonzero(np.array(self.J) == u)

            fsumx[u] = np.sum(-self.fnor[c1[0]] * self.nx[c1[0]]) + np.sum(
                self.fnor[c2[0]] * self.nx[c2[0]]) + np.sum(
                    self.ftan[c1[0]] *
                    (-self.ny[c1[0]])) + np.sum(-self.ftan[c2[0]] *
                                                (-self.ny[c2[0]]))
            fsumy[u] = np.sum(-self.fnor[c1[0]] * self.ny[c1[0]]) + np.sum(
                self.fnor[c2[0]] * self.ny[c2[0]]) + np.sum(
                    self.ftan[c1[0]] * self.nx[c1[0]]) + sum(
                        -self.ftan[c2[0]] * self.nx[c2[0]])
            torsum[u] = self.rad[u] * (np.sum(self.ftan[c1[0]]) +
                                       np.sum(self.ftan[c2[0]]))
            self.prepart[u] = np.sum(self.fnor[c1[0]] * self.rad[u]) + np.sum(
                self.fnor[c2[0]] * self.rad[u])
            self.sxxpart[u] = self.rad[u] * (
                np.sum(-self.fnor[c1[0]] * self.nx[c1[0]] * self.nx[c1[0]]) +
                np.sum(self.fnor[c2[0]] * self.nx[c2[0]] * (-self.nx[c2[0]])) +
                np.sum(self.ftan[c1[0]] * (-self.ny[c1[0]]) *
                       (-self.ny[c1[0]])) + np.sum(-self.ftan[c2[0]] *
                                                   (-self.ny[c2[0]]) *
                                                   (self.ny[c2[0]])))
            self.syypart[u] = self.rad[u] * (
                np.sum(-self.fnor[c1[0]] * self.ny[c1[0]] * (self.ny[c1[0]])) +
                np.sum(self.fnor[c2[0]] * self.ny[c2[0]] * (-self.ny[c2[0]])) +
                np.sum(self.ftan[c1[0]] * (self.nx[c1[0]]) * self.nx[c1[0]]) +
                sum(-self.ftan[c2[0]] * self.nx[c2[0]] * (-self.nx[c2[0]])))
            self.sxypart[u] = self.rad[u] * (
                np.sum(-self.fnor[c1[0]] * self.nx[c1[0]] *
                       (-self.ny[c1[0]])) +
                np.sum(self.fnor[c2[0]] * self.nx[c2[0]] * (self.ny[c2[0]])) +
                np.sum(self.ftan[c1[0]] * (-self.ny[c1[0]]) *
                       (self.nx[c1[0]])) + np.sum(-self.ftan[c2[0]] *
                                                  (-self.ny[c2[0]]) *
                                                  (-self.nx[c2[0]])))
            self.syxpart[u] = self.rad[u] * (
                np.sum(-self.fnor[c1[0]] * self.ny[c1[0]] * (self.nx[c1[0]])) +
                np.sum(self.fnor[c2[0]] * self.ny[c2[0]] *
                       (-self.nx[c2[0]])) + np.sum(self.ftan[c1[0]] *
                                                   (self.nx[c1[0]]) *
                                                   (-self.ny[c1[0]])) +
                sum(-self.ftan[c2[0]] * self.nx[c2[0]] * (self.ny[c2[0]])))
            del c1
            del c2

        # statistical output: don't do histograms here just yet; the problem of consistent and appropriate binning is too messy.
        # return the whole thing.
        # except for the mobilization distribution which is unique
        # just get a few signature parameters here
        zav = 2 * len(self.I) / (1.0 * self.N)
        nm = len(np.nonzero(self.fullmobi > 0)[0]) / (1.0 * self.N)
        pres = np.sum(self.prepart) / (self.Lx * self.Ly)
        sxx = np.sum(self.sxxpart) / (self.Lx * self.Ly)
        syy = np.sum(self.syypart) / (self.Lx * self.Ly)
        sxy = np.sum(self.sxypart) / (self.Lx * self.Ly)
        syx = np.sum(self.syxpart) / (self.Lx * self.Ly)
        fxbal = np.mean(abs(fsumx)) / np.mean(self.fnor)
        fybal = np.mean(abs(fsumy)) / np.mean(self.fnor)
        torbal = np.mean(abs(torsum)) / (
            np.mean(self.fnor) * np.mean(self.rad)
        )  # correct units; do *not* normalize with ftan; consider a mu though
        mobin = np.linspace(-1, 1, 101)
        mohist, bin_edges = np.histogram(self.ftan / (self.mu * self.fnor),
                                         mobin)

        return zav, nm, pres, fxbal, fybal, torbal, mobin, mohist, sxx, syy, sxy, syx

    #### =================== Computes the D2_min, amount of non-affine motion around a particle ==============
    def getD2min(self, threshold_rad):
        D2_min = np.zeros(len(self.x))
        for k in range(len(self.x)):
            # Boundary is dubious
            # the 10^5 is bad, but then they tend to be that value for experiment
            # just a way to get the color scale to not f up
            if (k in self.bindices):
                D2_min[k] = 1e5
            else:
                temp = []
                dist2 = (self.x - self.x[k])**2 + (self.y - self.y[k])**2
                rad2 = (self.rad[k] + np.mean(self.rad) + threshold_rad)**2
                # yes, that includes myself, but that term drops out
                temp = np.nonzero(dist2 < rad2)
                #for i in range(len(self.x)):
                #if np.sqrt((self.x[k]-self.x[i])**2+(self.y[k]-self.y[i])**2)<(self.rad[k]+threshold_rad):
                #temp.append(i)
                X = np.zeros((2, 2))
                Y = np.zeros((2, 2))
                epsilon = np.zeros((2, 2))
                if len(temp[0]) > 0:

                    for neighbor in temp[0]:
                        dx_next = self.xnext[neighbor] - self.xnext[k]
                        dy_next = self.ynext[neighbor] - self.ynext[k]
                        dx = self.x[neighbor] - self.x[k]
                        dy = self.y[neighbor] - self.y[k]
                        X[0, 0] += dx_next * dx
                        X[0, 1] += dx_next * dy
                        X[1, 0] += dy_next * dx
                        X[1, 1] += dy_next * dy
                        Y[0, 0] += dx * dx
                        Y[0, 1] += dx * dy
                        Y[1, 0] += dy * dx
                        Y[1, 1] += dy * dy
                    epsilon[0, 0] += (X[0, 0] / Y[0, 0] + X[0, 1] / Y[0, 1])
                    epsilon[0, 1] += (X[0, 0] / Y[1, 0] + X[0, 1] / Y[1, 1])
                    epsilon[1, 0] += (X[1, 0] / Y[0, 0] + X[1, 1] / Y[0, 1])
                    epsilon[1, 1] += (X[1, 0] / Y[1, 0] + X[1, 1] / Y[1, 1])

                    for neighbor in temp[0]:
                        dx_next = self.xnext[neighbor] - self.xnext[k]
                        dy_next = self.ynext[neighbor] - self.ynext[k]
                        dx = self.x[neighbor] - self.x[k]
                        dy = self.y[neighbor] - self.y[k]
                        D2_min[k] += (
                            (dx_next -
                             (epsilon[0, 0] * dx + epsilon[0, 1] * dy))**2 +
                            (dy_next -
                             (epsilon[1, 0] * dx + epsilon[1, 1] * dy))**2)

        return D2_min

    # ====== Experimental or simulation displacements, decomposed into normal, tangential and potentiallt rotational displecements
    def Disp2Contacts(self, minThresh, debug=False):
        disp2n = np.zeros(self.ncon)
        disp2t = np.zeros(self.ncon)
        if self.hasAngles:
            disp2r = np.zeros(self.ncon)
            disp2gear = np.zeros(self.ncon)
        for k in range(self.ncon):
            i = self.I[k]
            j = self.J[k]
            nx0 = self.nx[k]
            ny0 = self.ny[k]
            tx0 = -self.ny[k]
            ty0 = self.nx[k]
            disp2n[k] = ((self.dx[j] - self.dx[i]) * nx0 +
                         (self.dy[j] - self.dy[i]) * ny0)**2
            disp2t[k] = ((self.dx[j] - self.dx[i]) * tx0 +
                         (self.dy[j] - self.dy[i]) * ty0)**2
            if self.hasAngles:
                disp2r[0] = (self.rad[j] * self.dalpha[j] -
                             self.rad[i] * self.dalpha[i])**2
                disp2gear[0] = ((self.dx[j] - self.dx[i]) * tx0 +
                                (self.dy[j] - self.dy[i]) * ty0 -
                                (self.rad[i] * self.dalpha[i] +
                                 self.rad[j] * self.dalpha[j]))**2
        thresh = np.mean(disp2n + disp2t)
        print("Internally generated threshold of " + str(thresh))
        if thresh < minThresh:
            thresh = minThresh
        if self.hasAngles:
            return disp2n, disp2t, disp2r, disp2gear, thresh
        else:
            return disp2n, disp2t, thresh

    ##================ Finally, pure helper functions: positions of both ends of a contact ===============
    # get the cleaned up, periodic boundary conditions sorted out positions corresponding to two ends of a contact.
    # Basic plotting helper function
    def getConPos(self, k):
        x0 = self.x[self.I[k]]
        x1 = self.x[self.J[k]]
        y0 = self.y[self.I[k]]
        y1 = self.y[self.J[k]]
        if self.periodic:
            x1 = x1 - self.Lx * np.round((x1 - x0) / self.Lx)
            yover = np.round((y1 - y0) / self.Ly)
            if (yover != 0):
                y1 = y1 - self.Ly * yover
                x1 -= self.Lx * self.strain * yover
                x1 = x1 - self.Lx * np.round((x1 - x0) / self.Lx)
        if self.addBoundary:
            ival = self.I[k]
            if ((ival == self.bindices[0])
                    or (ival == self.bindices[1])):  #top or bottom
                x0 = x1
            if ((ival == self.bindices[2])
                    or (ival == self.bindices[3])):  #left or right
                y0 = y1
        return x0, x1, y0, y1

    # same, but based on existing particle labels (in case those come from elsewhere)
    def getConPos2(self, k1, k2):
        x0 = self.x[k1]
        x1 = self.x[k2]
        y0 = self.y[k1]
        y1 = self.y[k2]
        if self.periodic:
            x1 = x1 - self.Lx * np.round((x1 - x0) / self.Lx)
            yover = np.round((y1 - y0) / self.Ly)
            if (yover != 0):
                y1 = y1 - self.Ly * yover
                x1 -= self.Lx * self.strain * yover
                x1 = x1 - self.Lx * np.round((x1 - x0) / self.Lx)
        if self.addBoundary:
            if ((k1 == self.bindices[0])
                    or (k1 == self.bindices[1])):  #top or bottom
                x0 = x1
            if ((k1 == self.bindices[2])
                    or (k1 == self.bindices[3])):  #left or right
                y0 = y1
        return x0, x1, y0, y1
