class C(object):
    def __init__(self):
        self._x = None

    def getx(self):
        return self._x

    def setx(self, value):
        self._x = value

    def delx(self):
        del self._x

    x = property(getx, setx, delx, "I'm the 'x' property.")


import numpy as np

s=5
k=3
X=np.identity(s)
Y=np.zeros((s,k))
Z=np.concatenate((X,Y), axis=1)
W=np.insert(Z, 3, np.zeros(s),axis=1)
#print np.dot(W,W.T)
#print W

W = np.array([[1,5, 10],[3, 10, 18], [8, 5, 11]], dtype=float)
#print W
#eta = np.zeros(2)
#eta[1]=1
#R=np.dot(W,eta)
#print np.dot(R, eta.T)

Sigma=np.dot(W,W.T)
print Sigma
print np.linalg.inv(Sigma)
mu = np.random.normal(size=Sigma.shape[0])
j=0
mu[j]=0
print np.dot(Sigma[:,j].T, np.linalg.inv(Sigma))
print np.dot(np.dot(Sigma[:,j].T, np.linalg.inv(Sigma)), mu)
