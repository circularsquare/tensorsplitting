import sys
from pathlib import Path
sys.path.insert(1, str(Path.home()) + '/.exatn')
import exatn, numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from scipy.optimize import minimize, rosen, rosen_der

def conjugate(tName):
    exatn.createTensor(tName+'conj', np.conj(exatn.getLocalTensor(tName)))
def vsToMatrix(vs):
        v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, vi1, vi2, vi3, vi4, vi5, vi6, vi7, vi8, vi9, vi10, vi11, vi12 = vs[0], vs[1], vs[2], vs[3], vs[4], vs[5], vs[6], vs[7], vs[8], vs[9], vs[10], vs[11], vs[12], vs[13], vs[14], vs[15], vs[16], vs[17], vs[18], vs[19], vs[20], vs[21], vs[22], vs[23]
        return(np.array([[[v1, v2],[v2, v3]],[[v4, v5],[v5, v6]],[[v7,v8],[v8,v9]],[[v10,v11],[v11,v12]]]) + np.array([[[vi1, vi2],[-vi2, vi3]],[[vi4, vi5],[-vi5, vi6]],[[vi7,vi8],[-vi8,vi9]],[[vi10,vi11],[-vi11,vi12]]])*1j)
        
array1 = np.array([[0.4375+0.j, 0.    +0.j, 0.    +0.j, 0.375 +0.j],
        [0.    +0.j, 0.0625+0.j, 0.    +0.j, 0.    +0.j],
        [0.    +0.j, 0.    +0.j, 0.0625+0.j, 0.    +0.j],
        [0.375 +0.j, 0.    +0.j, 0.    +0.j, 0.4375+0.j]])
array2 = np.array([[0.390625+0.j, 0.      +0.j, 0.      +0.j, 0.28125 +0.j],
        [0.      +0.j, 0.109375+0.j, 0.      +0.j, 0.      +0.j],
        [0.      +0.j, 0.      +0.j, 0.109375+0.j, 0.      +0.j],
        [0.28125 +0.j, 0.      +0.j, 0.      +0.j, 0.390625+0.j]])

chosen = array2
chosen = np.reshape(chosen, (2,2,2,2))
chosen = np.transpose(chosen, (0,2,1,3))
print(np.reshape(chosen, (4,4)))
exatn.createTensor('rho', chosen.copy())

#isometryGuess = np.array([[[1,0],[0,0]],[[0,1/np.sqrt(2)],[1/np.sqrt(2),0]],[[0,0],[0,0]],[[0,0],[0,1]]]) + np.array([[[0,0],[0,0]],[[0,0],[0,0]],[[0,1/np.sqrt(2)],[-1/np.sqrt(2),0]],[[0,0],[0,0]]])*1j
#isometryGuess = np.array([[[1,0],[0,0]],[[0,.5],[.5,0]],[[0,.5],[.5,0]],[[0,0],[0,1]]]) + np.array([[[0,0],[0,0]],[[0,-.5],[.5,0]],[[0,.5],[-.5,0]],[[0,0],[0,0]]])*1j
# isometryGuess = np.array([[[1,0],[0,0]],[[0,1],[0,0]],[[0,0],[1,0]],[[0,0],[0,1]]]) + np.array([[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]]])*1j
isometryGuess = vsToMatrix([0.3921192815398409, 0.5246871481756136, -0.180443098223323, 0.4888572453246023, 0.0028485871220346897, -0.22495939396014603, -0.4888572453246018, -0.002848587122034066, 0.22495939396014578, -0.4399153380818597, 0.4740117800668652, 0.20243760074161518, 0, -0.36269101893135614, 0, 0.29559613643642724, 0.3243421309863568, 0.6423573181056297, 0.29559613643642757, -0.3243421309863568, 0.6423573181056297, 0, 0.3975670873211118, 0])
I = exatn.createTensor('I', isometryGuess)
conjugate('I')
#test I is isometry, II should be identity
exatn.createTensor('II', [4,4], 0+0j)
exatn.contractTensors('II(a,b)=I(a,i,j)*Iconj(b,i,j)')
#test I is symmetric, this maybe should be 0ish
#print(exatn.getLocalTensor('I') - np.transpose(exatn.getLocalTensor('Iconj'), (0, 2, 1)))

rho = np.reshape(chosen, (4,4))
w, sig1, e = np.linalg.svd(rho)[0], np.linalg.svd(rho)[1], np.linalg.svd(rho)[2]
w, e = np.reshape(w, (2,2,4)), np.reshape(e, (4,2,2))
print(sig1)
print(w)
print(e)

wi = np.tensordot(w.copy(), exatn.getLocalTensor('I').copy(), axes = ((2), (0)))
ei = np.tensordot(e.copy(), exatn.getLocalTensor('I').copy(), axes = ((0), (0)))

print('WI, EI')
print(wi)
print(ei)
print('differences')
print((wi - np.transpose(np.conjugate(wi), (1,0,3,2))).round(4))
print((ei - np.transpose(np.conjugate(ei), (1,0,3,2))).round(4))

wim = np.transpose(wi, (1,0,3,2)).reshape(4, 4)
eim = np.transpose(ei, (1,0,3,2)).reshape(4, 4)
svdw = np.linalg.svd(wim)
svde = np.linalg.svd(eim)
print(svdw[1].round(5))
print(svde[1].round(5))
print(svdw[0].round(3))
print(svdw[2].round(3))
print(svde[0].round(3))
print(svde[2].round(3))

def isomToError(vs):
        v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, vi1, vi2, vi3, vi4, vi5, vi6, vi7, vi8, vi9, vi10, vi11, vi12 = vs[0], vs[1], vs[2], vs[3], vs[4], vs[5], vs[6], vs[7], vs[8], vs[9], vs[10], vs[11], vs[12], vs[13], vs[14], vs[15], vs[16], vs[17], vs[18], vs[19], vs[20], vs[21], vs[22], vs[23]
        isometryGeneral = np.array([[[v1, v2],[v2, v3]],[[v4, v5],[v5, v6]],[[v7,v8],[v8,v9]],[[v10,v11],[v11,v12]]]) + np.array([[[vi1, vi2],[-vi2, vi3]],[[vi4, vi5],[-vi5, vi6]],[[vi7,vi8],[-vi8,vi9]],[[vi10,vi11],[-vi11,vi12]]])*1j

        isomDoubled = np.tensordot(isometryGeneral, isometryGeneral.conjugate(), axes=([1,2],[1,2]))
        isomError = isomDoubled-np.eye(4)

        #print(isometryGeneral)
        #print(exatn.getLocalTensor('W').transpose(2,0,1))
        Wisom = np.tensordot(w, isometryGeneral, axes = ((2), (0)))
        Eisom = np.tensordot(e, isometryGeneral, axes = ((0), (0)))

        #Wisom = np.einsum('ijk,klm->ijlm', exatn.getLocalTensor('W'), isometryGeneral)
        #print(Wisom)
        Wisomconj = np.conjugate(np.transpose(Wisom, (1,0,3,2)))
        Eisomconj = np.conjugate(np.transpose(Eisom, (1,0,3,2)))
        symErrorW = Wisom - Wisomconj 
        symErrorE = Eisom - Eisomconj
        #outError = np.sum(np.square(symError)) + np.sum(np.square(isomError))
        outError = np.linalg.norm(symErrorW) + np.linalg.norm(symErrorE) + np.linalg.norm(isomError)*2
        if np.random.rand() < .001:
                print(np.linalg.norm(symErrorW) , np.linalg.norm(symErrorE) , np.linalg.norm(isomError))
        return outError

x0 = np.array([0.3921192815398409, 0.5246871481756136, -0.180443098223323, 0.4888572453246023, 0.0028485871220346897, -0.22495939396014603, -0.4888572453246018, -0.002848587122034066, 0.22495939396014578, -0.4399153380818597, 0.4740117800668652, 0.20243760074161518, 0, -0.36269101893135614, 0, 0.29559613643642724, 0.3243421309863568, 0.6423573181056297, 0.29559613643642757, -0.3243421309863568, 0.6423573181056297, 0, 0.3975670873211118, 0])
# x0 = np.random.rand(24)
res = minimize(isomToError, x0, method='nelder-mead', tol=1e-13, options={'maxiter': 10000000, 'disp': True})
print(res['x'].tolist())
print(res['fun'])

# results for  array1 = np.array([[0.4375+0.j, 0.    +0.j, 0.    +0.j, 0.375 +0.j],
#         [0.    +0.j, 0.0625+0.j, 0.    +0.j, 0.    +0.j],
#         [0.    +0.j, 0.    +0.j, 0.0625+0.j, 0.    +0.j],
#         [0.375 +0.j, 0.    +0.j, 0.    +0.j, 0.4375+0.j]])
# W and E are the same up to transpose. NW is:
# [[ 0.191-0.j     0.691-0.j    -0.518-0.j    -0.467-0.j   ]
#  [-0.463+0.335j  0.398-0.023j  0.024+0.458j  0.374-0.405j]
#  [-0.568-0.34j   0.168+0.141j  0.486-0.001j -0.523+0.07j ]
#  [ 0.21 +0.393j -0.073+0.557j  0.138+0.517j -0.175+0.412j]]
# SW is its negative transpose, and NE and SE are the same. 
# the singular valuse are  [1.41375 1.41374 0.03552 0.03529]
# [1.41375 1.41374 0.03552 0.03529], promising

# results for array2 = array2 = np.array([[0.390625+0.j, 0.      +0.j, 0.      +0.j, 0.28125 +0.j],
#         [0.      +0.j, 0.109375+0.j, 0.      +0.j, 0.      +0.j],
#         [0.      +0.j, 0.      +0.j, 0.109375+0.j, 0.      +0.j],
#         [0.28125 +0.j, 0.      +0.j, 0.      +0.j, 0.390625+0.j]])
# get isometry :
# [0.3921192815398409, 0.5246871481756136, -0.180443098223323, 0.4888572453246023, 0.0028485871220346897, -0.22495939396014603, -0.4888572453246018, -0.002848587122034066, 0.22495939396014578, -0.4399153380818597, 0.4740117800668652, 0.20243760074161518, 0, -0.36269101893135614, 0, 0.29559613643642724, 0.3243421309863568, 0.6423573181056297, 0.29559613643642757, -0.3243421309863568, 0.6423573181056297, 0, 0.3975670873211118, 0])
# W and E are the same... but NW and SW are not quite transposes (and so arent NE and SE)
# singular values are [1,1,1,1], weird