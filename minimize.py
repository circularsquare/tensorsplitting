import sys
from pathlib import Path
sys.path.insert(1, str(Path.home()) + '/.exatn')
import exatn, numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from scipy.optimize import minimize, rosen, rosen_der


array1 = np.array([[0.4375+0.j, 0.    +0.j, 0.    +0.j, 0.0625+0.j],
        [0.    +0.j, 0.    +0.j, 0.375 +0.j, 0.    +0.j],
        [0.    +0.j, 0.375 +0.j, 0.    +0.j, 0.    +0.j],
        [0.0625+0.j, 0.    +0.j, 0.    +0.j, 0.4375+0.j]])
array2 = np.array([[0.3828125+0.j, 0.       +0.j, 0.       +0.j, 0.0546875+0.j],
        [0.       +0.j, 0.28125  +0.j, 0.046875 +0.j, 0.       +0.j],
        [0.       +0.j, 0.046875 +0.j, 0.28125  +0.j, 0.       +0.j],
        [0.0546875+0.j, 0.       +0.j, 0.       +0.j, 0.3828125+0.j]])

array1 = np.array([[0.4375+0.j, 0.    +0.j, 0.    +0.j, 0.375 +0.j],
        [0.    +0.j, 0.0625+0.j, 0.    +0.j, 0.    +0.j],
        [0.    +0.j, 0.    +0.j, 0.0625+0.j, 0.    +0.j],
        [0.375 +0.j, 0.    +0.j, 0.    +0.j, 0.4375+0.j]])
array2 = np.array([[0.390625+0.j, 0.      +0.j, 0.      +0.j, 0.28125 +0.j],
        [0.      +0.j, 0.109375+0.j, 0.      +0.j, 0.      +0.j],
        [0.      +0.j, 0.      +0.j, 0.109375+0.j, 0.      +0.j],
        [0.28125 +0.j, 0.      +0.j, 0.      +0.j, 0.390625+0.j]])

chosen = array1
chosen = np.reshape(chosen, (2,2,2,2))
chosen = np.transpose(chosen, (0,2,1,3)) # now 0,1,2,3 correspond to nw,sw,ne,se

#  --nw---ne--
#          |
#  --sw---se--
# (all bonds 2d)
def findError(v):
    nw = np.array([[v[0],v[1]],[v[2],v[3]]]) + np.array([[v[4],v[5]],[v[6],v[7]]])*1j
    sw = np.conjugate(nw) #is a transpose necessary here? 
    ne = np.array([[[v[8],v[9]],[v[10],v[11]]],[[v[12],v[13]],[v[14],v[15]]]]) + np.array([[[v[16],v[17]],[v[18],v[19]]],[[v[20],v[21]],[v[22],v[23]]]])*1j
    se = np.conjugate(np.transpose(ne,(0,2,1)))
    testW = np.tensordot(nw, sw, axes=0) # indices 0 and 1 correspond to nw, 2 and 3 for sw
    testE = np.tensordot(ne, se, axes=((2), (2))) # 0 and 1 correspond to ne, 2 and 3 for se
    testRho = np.tensordot(testW, testE, axes=((0,2),(0,2))) # 0,1,2,3 corespond to nw,sw,ne,se
    return np.linalg.norm(chosen-testRho)

x0 = np.random.rand(24)
#print(findError(x0))
res = minimize(findError, x0, method='bfgs', tol=1e-10, options={'maxiter': 100000, 'disp': True})
print(res['x'])
print(res['fun'])




'''
todo:
make the internal bond 4d
'''