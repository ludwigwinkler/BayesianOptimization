import numpy as np
import matplotlib.pylab as plt
import math
import random



def kernel(a,b,para):
        return math.exp(-(a-b)**2/(2*para))


def sample(func,noise):
        x = (random.random()-.5)*10
        y = func(x) + noise(0,0.01)
        return [x,y]

def kernelMat(a,b,para):
        mat = np.zeros((len(a),len(b)))
        for ai in range(len(a)):
                for bi in range(len(b)):
                        mat[ai,bi] = kernel(a[ai],b[bi],para)
        return mat


def linear(x):
        return math.sin(x)

def probZ(z,ozz,ozs,y):
        #print("z: %f vs zy:%f" %(z**2 * ozz,z * ozs * y))

        try:

                return math.exp(-0.5*(z**2 * ozz) - z * ozs * y)
        except OverflowError:
                print("overflow")
                return 0


sampleN = 3
kernelP = 0.1
count = 0
samples = np.zeros((2,sampleN))
while(count < sampleN):
        nsample = sample(linear,np.random.normal)
        samples[:,count] = nsample
        count += 1

ys = np.matrix(samples[1,:]).T

kss = kernelMat(samples[0,:],samples[0,:],kernelP)

curMat = np.zeros((sampleN+1,sampleN+1))
curMat[1:,1:] = kss

imMat = []
ranges = (-7,7,-4.5,4.5)
zopt = []
zoptx= []
for xi in np.arange(ranges[0],ranges[1],0.2):
        imMat.append([])
        kzz = kernel(xi,xi,kernelP)
        ksz = kernelMat([xi],samples[0,:],kernelP)
        curMat[0,0] = kzz
        curMat[0,1:] = ksz
        curMat[1:,0] = ksz


        omega = np.linalg.inv(curMat)

        omega /= (omega.max() - omega.min())

        ozz = omega[0,0]
        ozs = np.matrix(omega[0,1:])
        zo = - (ozs * ys) / ozz

        zoptx.append(xi)
        zopt.append(- (ozs * ys) / ozz)

        for yi in np.arange(ranges[2],ranges[3],0.2):
                #print(" x: %f, y: %f " %(xi,yi))
                imMat[-1].insert(0,probZ(yi,ozz,ozs,ys))


zopt = np.array(zopt).flatten()
imMat = np.matrix(imMat).T
imrealM = np.matrix((imMat))
#for i in range(len(imMat)):
#  imrealM[i,:] /= imrealM[i,:].mean()
plt.imshow(imrealM,interpolation='nearest',extent=ranges)
plt.axis('on')
plt.colorbar()
plt.plot(zoptx,zopt,"g")
plt.plot(samples[0,:],samples[1,:],"rx")
plt.ylim([-4,4])
plt.show()