# SVDs a two site positive density operator into two parts,
# then finds an isometry/unitary that can be inserted between the parts
# that when conracted with them yields a hermitian product
# with the goal of being able to svd those products to get a canonical form
# of a four tensor network
import sys
from pathlib import Path
sys.path.insert(1, str(Path.home()) + '/.exatn')
import exatn, numpy as np
from scipy.optimize import minimize, rosen, rosen_der
from matplotlib import pyplot as plt
import pylab
import pickle
import os
import qtn

def conjugate(tName):
    exatn.createTensor(tName+'conj', np.conj(exatn.getLocalTensor(tName)))
def anglesToTensor(angles):
        return np.reshape(np.linalg.multi_dot([np.kron(SU2(angles[0:3]),SU2(angles[3:6])),SO3(angles[6:9]),np.kron(SU2(angles[9:12]),SU2(angles[12:15]))]), (4,2,2))
XX = np.tensordot(qtn.sx, qtn.sx,((),())).swapaxes(1,2).reshape(4,4)
YY = np.tensordot(qtn.sy, qtn.sy,((),())).swapaxes(1,2).reshape(4,4)
ZZ = np.tensordot(qtn.sz, qtn.sz,((),())).swapaxes(1,2).reshape(4,4)
SU2 = lambda a: np.linalg.multi_dot([qtn.Z(a[0]),qtn.Y(a[1]),qtn.Z(a[2])])
SO3 = lambda a: np.linalg.multi_dot([qtn.R(XX,a[0]),qtn.R(YY,a[1]),qtn.R(ZZ,a[2])])

arrays = [
np.array([[0.4375+0.j, 0.    +0.j, 0.    +0.j, 0.375 +0.j],
        [0.    +0.j, 0.0625+0.j, 0.    +0.j, 0.    +0.j],
        [0.    +0.j, 0.    +0.j, 0.0625+0.j, 0.    +0.j],
        [0.375 +0.j, 0.    +0.j, 0.    +0.j, 0.4375+0.j]]),
np.array([[0.390625+0.j, 0.      +0.j, 0.      +0.j, 0.28125 +0.j],
        [0.      +0.j, 0.109375+0.j, 0.      +0.j, 0.      +0.j],
        [0.      +0.j, 0.      +0.j, 0.109375+0.j, 0.      +0.j],
        [0.28125 +0.j, 0.      +0.j, 0.      +0.j, 0.390625+0.j]]),
np.array([[0.475+0.j, 0.   +0.j, 0.   +0.j, 0.45 +0.j],
        [0.   +0.j, 0.025+0.j, 0.   +0.j, 0.   +0.j],
        [0.   +0.j, 0.   +0.j, 0.025+0.j, 0.   +0.j],
        [0.45 +0.j, 0.   +0.j, 0.   +0.j, 0.475+0.j]] ),
np.array([[0.4525+0.j, 0.    +0.j, 0.    +0.j, 0.405 +0.j],
        [0.    +0.j, 0.0475+0.j, 0.    +0.j, 0.    +0.j],
        [0.    +0.j, 0.    +0.j, 0.0475+0.j, 0.    +0.j],
        [0.405 +0.j, 0.    +0.j, 0.    +0.j, 0.4525+0.j]])]

choice = 3 #change this to pick a different array to try
chosen = arrays[choice]
chosen = np.reshape(chosen, (2,2,2,2))
chosen = np.transpose(chosen, (0,2,1,3))
print(np.reshape(chosen, (4,4)))

#isometryGuess = np.array([[[1,0],[0,0]],[[0,1/np.sqrt(2)],[1/np.sqrt(2),0]],[[0,0],[0,0]],[[0,0],[0,1]]]) + np.array([[[0,0],[0,0]],[[0,0],[0,0]],[[0,1/np.sqrt(2)],[-1/np.sqrt(2),0]],[[0,0],[0,0]]])*1j
#isometryGuess = np.array([[[1,0],[0,0]],[[0,.5],[.5,0]],[[0,.5],[.5,0]],[[0,0],[0,1]]]) + np.array([[[0,0],[0,0]],[[0,-.5],[.5,0]],[[0,.5],[-.5,0]],[[0,0],[0,0]]])*1j
# isometryGuess = np.array([[[1,0],[0,0]],[[0,1],[0,0]],[[0,0],[1,0]],[[0,0],[0,1]]]) + np.array([[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]]])*1j
angles = [np.random.uniform(0, np.pi) for i in range(15)]
SSU = anglesToTensor(angles)
SSU = anglesToTensor([-0.020491765812750733, 1.0849813938928623, 0.1823094915565751, -0.01892176877947449, 1.8492804753843204, 0.16753058604914656, 1.5707963267913112, -7.266906333088912e-13, 1.5707963267988387, 2.4032056784753717, 0.13338369000216244, 0.5760507158714833, 0.28699706946011383, 0.3217444561240923, -0.11511191774971147])
I = exatn.createTensor('I', np.reshape(SSU, (4,2,2)))
conjugate('I')
#test I is isometry, II should be identity
exatn.createTensor('II', [4,4], 0+0j)
exatn.contractTensors('II(a,b)=I(a,i,j)*Iconj(b,i,j)')
print('I doubled')
exatn.print('II')

rho = np.reshape(chosen, (4,4))
w, sig1, e = np.linalg.svd(rho)[0], np.linalg.svd(rho)[1], np.linalg.svd(rho)[2]
w, e = np.reshape(w, (2,2,4)), np.reshape(e, (4,2,2))

wi = np.tensordot(w.copy(), exatn.getLocalTensor('I').copy(), axes = ((2), (0)))
ei = np.tensordot(e.copy(), exatn.getLocalTensor('I').copy(), axes = ((0), (0)))

print('WI')
print(wi)
print('differences')
print(np.linalg.norm(wi - np.transpose(np.conjugate(wi), (1,0,3,2))))
print(np.linalg.norm(ei - np.transpose(np.conjugate(ei), (1,0,3,2))))

wim = np.transpose(wi, (0,2,1,3)).reshape(4, 4)
eim = np.transpose(ei, (0,2,1,3)).reshape(4, 4)
svdw = np.linalg.svd(wim)
svde = np.linalg.svd(eim)
print('sigmaw, sigmae, nw, sw, ne, se:')
print(svdw[1].round(5))
print(svde[1].round(5))
print(svdw[0].round(3))
print(svdw[2].round(3))
print(svde[0].round(3))
print(svde[2].round(3))

def isomToError(angles):
        SSU = np.reshape(np.linalg.multi_dot([np.kron(SU2(angles[0:3]),SU2(angles[3:6])),SO3(angles[6:9]),np.kron(SU2(angles[9:12]),SU2(angles[12:15]))]), (4,2,2))

        isomDoubled = np.tensordot(SSU, SSU.conjugate(), axes=([1,2],[1,2]))
        isomError = isomDoubled-np.eye(4)

        #print(isometryGeneral)
        #print(exatn.getLocalTensor('W').transpose(2,0,1))
        Wisom = np.tensordot(w, SSU, axes = ((2), (0)))
        Eisom = np.tensordot(e, SSU, axes = ((0), (0)))

        #Wisom = np.einsum('ijk,klm->ijlm', exatn.getLocalTensor('W'), isometryGeneral)
        #print(Wisom)
        
        Wisomconj = np.conjugate(np.transpose(Wisom, (1,0,3,2)))
        Eisomconj = np.conjugate(np.transpose(Eisom, (1,0,3,2)))
        symErrorW = Wisom - Wisomconj 
        symErrorE = Eisom - Eisomconj
        outError = np.linalg.norm(symErrorW) + np.linalg.norm(symErrorE) + np.linalg.norm(isomError)*2
        if np.random.rand() < .0001:
                print(np.linalg.norm(symErrorW) , np.linalg.norm(symErrorE) , np.linalg.norm(isomError))
        return outError

x0 = np.random.rand(15)
res = minimize(isomToError, x0, method='nelder-mead', tol=1e-10, options={'maxiter': 1000000, 'disp': True})
print('v0 and the function')
print(res['x'].tolist())
print(res['fun'])
print(anglesToTensor(res['x'].tolist()))

# some working v0s are
# array0 :[-0.984149992376034, 1.6318994764366724e-06, 0.9841499923753997, 0.9274513894910155, -6.60743265519596e-14, 0.31246528607109825, 1.5707967683161328, 8.543707879425112e-07, 0.33087965123298396, 6.580595428436899, 0.7089187379617019, 0.07186405188026865, 1.324567662023722, -2.2426438923809666, -3.597222805113402]
# array1: [1.1295679270234313, 0.4233318417019411, 0.0074169434190576216, -0.11438257225062412, 0.14572513909547785, 0.21869398984198227, -1.2679165204638638e-11, 8.299802719693693e-12, 3.141592653592096, 0.7627013856589444, 0.2399283659126255, 0.5483819903066084, -0.5059276547809515, 0.43589611332949674, 1.121527389016972]
#array2: [0.08235698530077154, 0.09438824030108445, 1.2733026604950344, 1.0204526470264237, 0.12358066351798837, 0.28198929321025684, 3.967237652788242e-14, 4.068094273875353e-13, 3.141592653589382, -0.0758000903533961, 0.3979297153551528, 0.38262648223883994, 0.19262785945661398, 0.4793892177570186, 0.015345551652562482]
# array3: [0.033821988905258546, 0.012051901432242617, -0.32801430780247093, 1.6963894964269282, 3.07371010969484, 1.4017931342347079, -1.170963351703602e-11, 1.5707963267965779, 1.5707963267750913, 0.83348290683803, -0.3812988716164766, 0.41121922655883025, -0.6962480271762403, -0.33214702694364406, -0.5575980499926025]
# singular values are 
# array0: [1,1,1,1]
# array1: [1.70711 0.70711 0.70711 0.29289]
# array2: [1.22474 1.22474 0.70711 0.70711]
# array3: [1.00807 0.99997 0.99997 0.99193]
# also array3: [1.70711 0.70711 0.70711 0.29289]


# x0 = np.array([0.3921192815398409, 0.5246871481756136, -0.180443098223323, 0.4888572453246023, 0.0028485871220346897, -0.22495939396014603, -0.4888572453246018, -0.002848587122034066, 0.22495939396014578, -0.4399153380818597, 0.4740117800668652, 0.20243760074161518, 0, -0.36269101893135614, 0, 0.29559613643642724, 0.3243421309863568, 0.6423573181056297, 0.29559613643642757, -0.3243421309863568, 0.6423573181056297, 0, 0.3975670873211118, 0])
# for j in range(100):
#         x0 = np.random.rand(24)
#         res = minimize(isomToError, x0, method='nelder-mead', tol=1e-15, options={'maxiter': 1000000, 'disp': True})
#         print('v0 and the function')
#         print(res['x'].tolist())
#         print(res['fun'])
#         #print(vsToMatrix(res['x'].tolist()))

#         if res['fun']<.0001:
#                 isoms=[]
#                 filename = './isoms' + str(choice)
#                 if os.path.exists(filename):
#                         with open(filename,'rb') as rfp: 
#                                 isoms = pickle.load(rfp)
#                 isoms.append(vsToMatrix(res['x'].tolist()))
#                 print('isoms is:')
#                 print(isoms)
#                 print(len(isoms))
#                 with open(filename,'wb') as wfp:
#                         pickle.dump(isoms, wfp)

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
# [[[-0.39878348-1.45989204e-06j  0.02404137+6.21774472e-01j]
#   [ 0.02404137-6.21774472e-01j  0.25808673-2.85092641e-05j]]

#  [[-0.51880849+3.84187844e-01j -0.06749083-2.33441330e-01j]
#   [-0.06749083+2.33441330e-01j  0.3357647 +5.93632676e-01j]]

#  [[ 0.51880581+3.84190702e-01j  0.0674627 +2.33453846e-01j]
#   [ 0.0674627 -2.33453846e-01j -0.33576287+5.93630769e-01j]]

#  [[-0.08629772-5.35419843e-05j  0.70022555-6.63394640e-02j]
#   [ 0.70022555+6.63394640e-02j  0.05584931+1.11385022e-04j]]]
# W and E are the same... but NW and SW are not quite transposes (and so arent NE and SE)
# singular values are [1,1,1,1], weird


#tensor3: 
# [[[ 0.83018729-3.25360266e-15j -0.06250283+3.48532103e-01j]
#   [-0.06250283-3.48532103e-01j  0.24500326+2.97936706e-15j]]

#  [[ 0.01773878-2.00146085e-01j  0.49546739+6.58864957e-02j]
#   [ 0.49546739-6.58864957e-02j  0.00523503+6.78189903e-01j]]

#  [[-0.01773878-2.00146085e-01j -0.49546739-6.58864957e-02j]
#   [-0.49546739+6.58864957e-02j -0.00523503+6.78189903e-01j]]

#  [[ 0.47962779+7.21256608e-15j  0.07153689-6.08147443e-01j]
#   [ 0.07153689+6.08147443e-01j  0.14154682-2.24635448e-16j]]]

# another tensor3: 
# [[[-0.18376606+1.99245734e-04j  0.0151954 +6.94073356e-01j]
#   [ 0.0151954 -6.94073356e-01j -0.04788081+1.58519493e-04j]]

#  [[-0.02176535-1.78268564e-01j  0.49954921-1.40578331e-02j]
#   [ 0.49954921+1.40578331e-02j -0.00565824+6.84266515e-01j]]

#  [[ 0.0213235 -1.78295071e-01j -0.49956372+1.39068469e-02j]
#   [-0.49956372-1.39068469e-02j  0.00556951+6.84259142e-01j]]

#  [[ 0.94959584-7.80139339e-04j  0.02560807+1.33682555e-01j]
#   [ 0.02560807-1.33682555e-01j  0.24741308+2.13910623e-04j]]]

# another one: 
# [[[-0.01642986+2.68637457e-08j  0.67802294-1.96181166e-01j]
#   [ 0.67802294+1.96181166e-01j  0.05766917-4.03426496e-08j]]

#  [[ 0.19265844+6.80046459e-01j  0.04081276+3.35931701e-02j]
#   [ 0.04081276-3.35931701e-02j -0.67623543+1.93744198e-01j]]

#  [[-0.19265843+6.80046462e-01j -0.04081278-3.35932185e-02j]
#   [-0.04081278+3.35932185e-02j  0.67623543+1.93744187e-01j]]

#  [[ 0.02385662-1.08888748e-07j -0.19223299-6.77684252e-01j]
#   [-0.19223299+6.77684252e-01j -0.08373726+2.61381086e-08j]]]