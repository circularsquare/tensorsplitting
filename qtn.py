# Quantum Tensor Network (qtn) library
# Copyright: Eugene Dumitrescu (2016 - Present)

# Dependencies
import itertools
import networkx as nx
import numpy as np
import scipy.linalg
import scipy.sparse.linalg as ssl

######################################
####### Basic Quantum Tensors  #######
######################################

# primative gates and basic operators
sx = np.array([[0,1],[1,0]])
sy = np.array([[0,-1j],[1j,0]])
sz = np.array([[1,0],[0,-1]])
s0 = np.eye(2)

H = (sx + sz)/np.sqrt(2)
S = np.array([[1,0],[0,np.exp(1j*np.pi/2)]])
T = np.array([[1,0],[0,np.exp(1j*np.pi/4)]])

# projector and ladder operators
P0 = (s0+sz)/2
P1 = (s0-sz)/2
sp = (sx+1j*sy)/2
sm = (sx-1j*sy)/2

# parameterized gates
def psi1qb(theta, phi):
    """ rotate single qubit by:
    theta: polar angle
    phi: azimuthal angle
    """
    return np.array([np.cos(theta),np.exp(1j * phi) * np.sin(theta)])

def X(phi = None):
    """
    Rotation about X-axis by angle phi
    """
    if phi == None:
        return sx
    else:
        return scipy.linalg.expm(-1j * phi / 2 * sx)

def Y(phi = None):
    """
    Rotation about Y-axis by angle phi
    """
    if phi == None:
        return sy
    else:
        return scipy.linalg.expm(-1j * phi / 2 * sy)

def Z(phi = None):
    """
    Rotation about Z-axis by angle phi
    I.e. a phi phase gate
    """
    if phi == None:
        return sz
    else:
        return scipy.linalg.expm(-1j * phi / 2 * sz)

    
def R(op, phi = None):
    """
    Rotation about user designated operator by angle phi
    """
    if phi == None:
        return None
    else:
        return scipy.linalg.expm(-1j * phi / 2 * op)

# pauli matrix eigenstates
psi0 = np.array([1,0])
psi1 = np.array([0,1])
psip = np.array([1,1]) / np.sqrt(2)
psim = np.array([1,-1])/ np.sqrt(2)
psipi = np.array([1,1j]) / np.sqrt(2)
psimi = np.array([1,-1j])/ np.sqrt(2)

        #################
### 0 ###               ### 2 ###
        #    2-BODY     #
        #    INDEX      #
### 1 ###  CONVENTION   ### 3 ###
        #               #
        #################

# CNOT-like gates
CX = np.kron(P0,s0) + np.kron(P1,sx)
CX.shape = (2,2,2,2)
CZ = np.diag([1,1,1,-1])
CZ.shape = (2,2,2,2)
SWAP = np.real((np.kron(s0,s0) + np.kron(sx,sx)
     + np.kron(sy,sy) + np.kron(sz,sz))/2)
SWAP.shape = (2,2,2,2)

# Bell states
B0 = np.tensordot(psip, np.tensordot(psi0, CX, axes = ([0,1])), axes = ([0,0]))
Bx = np.tensordot(B0, sx, axes = ([0,0]))
By = 1j * np.tensordot(B0, sy, axes = ([0,0]))
Bz = np.tensordot(B0, sz, axes = ([0,0]))

############ controlled multi-qubit operations ############

def Controlled(U):
    """
    Generalized controlled unitary tensor construction
    currently only works for gates on one target qubit -B
    """
    shp = U.shape
    new_ten = scipy.linalg.block_diag(np.eye(*shp), U)
    return new_ten.reshape(2, shp[0], 2, shp[1])

def Controlled2(U):
    """
    Generalized controlled unitary tensor construction
    this one only works for gates on two target qubits -B
    """
    '''Generalized controlled unitary tensor construction
    Parameters:
    -----------
    U: input tensor which is assumed to be a square Matrix

    Returns:
    --------
    Controlled unitary

    '''
    shp = U.shape
    new_ten = scipy.linalg.block_diag(np.eye(*shp), U)
    return new_ten.reshape(2, shp[0], 2, shp[1], 2, shp[2])

def CR(phi):
    """ controlled phase gates of arbitrary angle """
    return (np.kron(P0,s0) + np.kron(P1,R(phi)))

def CRS(site):
    """ Controlled phase rotation followed by SWAP gate.
    The angle of rotation is determined by the site index :== pi/2**(site)"""
    return np.dot(CR(np.pi/2**(site)),SWAP)

#############################################
###### (Tree) Tensor Network Functions ######
#############################################

def shared_legs(val, leg_list):
    """
    Parameters
    ----------
    val: leg number
    leg_list: list of legs associated connected to each tensor

    Returns
    ------
    sout: connecting tensors and legs eminating from tensors
    """
    out = [list(legs) for legs in leg_list if legs.__contains__(val)]
    sout = sum(out,[])
    if len(out) == 1:
        sout.remove(val)
    if len(out) == 2:
        sout.remove(val)
        sout.remove(val)
    return sout

def connectors(l_lin, legs_list, nt_allowed):
    """
    Parameters
    ----------
    l_lin:
    legs_list:
    nt_allowed:

    """
    a = [legs for legs in legs_list if legs.__contains__(l_lin)]
    merged = list(itertools.chain.from_iterable(a))   # flatten
    nt_all = nt_allowed + [l_lin]                     # connectors + input
    return [i for i in merged  if i>0 and not(nt_all.__contains__(i))] # remove negatives # and nt_allowed

def heuristic_path(l_s, l_f, legs_list, candidates = []):
    """
    Parameters
    ----------
    l_s: inital leg to begin seach
    l_f: final leg
    legs_list: list of legs budding from tensors

    Returns
    -------
    path: list of legs connecting leg_from to leg_to
    """
    candy =  candidates
    leg_path = [l_s + [c] for c in connectors(l_s[-1], legs_list, l_s)]
    for l in leg_path:
    	if l_f in l:
    		candidates.append(l)
    	else:
    		heuristic_path(l, l_f, legs_list, candy)
    return candidates

def path(l_s, l_f, leg_list):
    """
    return minimum length candidate heuristic_path
    """
    # candidates =
    return min(heuristic_path([l_s], l_f, leg_list, []), key = len)

def leg_find(val, leg_list):
    """
    val: leg number
    leg_list: list of legs associated connected to each tensor
    out: single (pair of) tensor(s) connected to val
    """
    return [[leg_list.index(legs),legs.index(val)] for legs in leg_list if legs.__contains__(val)]

def leg_tr(tr_leg, tens_list, legs_list, ent_list):
    """trace over the specified leg
    tr_leg: leg to be traced over (has to appear twice on a tensor) """
    q_index = [legs.__contains__(tr_leg) for legs in legs_list].index(True,0)
    ax1 = legs_list[q_index].index(tr_leg,0)
    ax2 = legs_list[q_index].index(tr_leg,ax1+1)
    tens_list[q_index] = np.trace(tens_list[q_index], offset=0, axis1=ax1, axis2=ax2)
    legs_list[q_index].remove(tr_leg)
    legs_list[q_index].remove(tr_leg)
    ent_list[tr_leg] = np.array([0])

def split(input_legs, tensor_list, leg_list, ent_list, cutoff):
    """ Function splitting input_legs off a tensor by SVD.
    Also updates tensor, leg, and Schmidt lists."""
    # find tensor common to all input legs
    input_inds = [set([leg_list.index(legs) for legs in leg_list
     if legs.__contains__(qs)]) for qs in input_legs]
    ind = list(set.intersection(*input_inds))[0]
    svd_ten = tensor_list[ind]        # tensor to decompose
    svd_legs = leg_list[ind]          # all tensor legs
    start_shape = svd_ten.shape       # index dimensions
    # permute tensor with leg and index data
    top_ind = [svd_legs.index(qs) for qs in input_legs]
    top_ind.sort()   # sort input indicies for consistency
    others = list(svd_legs)
    [others.remove(qs) for qs in input_legs]
    bot_ind = [svd_legs.index(legs) for legs in others]
    top_legs = [svd_legs[i] for i in top_ind]
    bot_legs = [svd_legs[i] for i in bot_ind]
    # Search for unused leg number
    emptys = [elements.any() == 0 for elements in ent_list[1:]]
    if any(emptys):
        new_leg = emptys.index(True)+1
    else:
        new_leg = max(list(itertools.chain.from_iterable(leg_list)))+1
    ##### permute legs of svd_ten and update svd_inds as we go along
    top_js = list(top_ind)
    top_js.sort(reverse = True)
    svd_ind = list(range(len(start_shape))) # number all tensor legs
    for element in top_js:
        for i in list(range(svd_ind.index(element),0,-1)):
            svd_ind[i], svd_ind[i-1] = svd_ind[i-1], svd_ind[i]
            svd_ten = svd_ten.swapaxes(i,i-1)
    # outgoing leg dimensions
    top_dim = np.prod([start_shape[top_ind[i]] for i in range(len(top_ind))])
    bot_dim = np.prod([start_shape[bot_ind[i]] for i in range(len(bot_ind))])
    # used to shape post SVD U and V
    top_shape = [start_shape[shapes] for shapes in top_ind]
    bot_shape = [start_shape[shapes] for shapes in bot_ind]
    # singular value decomposition
    F = svd_ten.reshape(top_dim, bot_dim)
    # print(top_dim, bot_dim, input_legs)
    # pass statement fixing top_dim = 0 ??
    if top_dim == 0 or bot_dim ==0:
        print('This should not happen!!!')
        pass
    if top_dim * bot_dim < 1024:
        U, S, V = np.linalg.svd(F)
        U, S, V = trun(U,S,V,cutoff)
    else:
        if (top_dim==bot_dim):
            # expand both matrux dimensions by 1
            print('dimensionality equality')
            w = np.eye(top_dim,top_dim+1)
            m = np.tensordot(np.tensordot(w,F,[[0],[0]]),w,[[1],[0]])
            U, S, V = ssl.svds(m, top_dim)
            # undo isometry on the decomposed tensors
            U = np.tensordot(w,U,[[1],[0]])
            V = np.tensordot(w,V,[[1],[1]])
        else:
            #expand smaller dimension, be it the top (U) or bottom (V)
            m_dim = min(top_dim, bot_dim) # smallest dimension
            w = np.eye(m_dim,m_dim+1)     # dimension increasing isometry
            if m_dim == top_dim:
                U, S, V = ssl.svds(np.tensordot(w,F,[[0],[0]]), m_dim)  # sparse SVD
                U = np.tensordot(w,U,[[1],[0]])
            else:
                U, S, V = ssl.svds(np.tensordot(w,F,[[0],[1]]).transpose()
                , m_dim)  # sparse SVD
                print(V.shape, w.shape)
                V = np.tensordot(w,V,[[1],[1]])
    # print('SVD done. New virtual leg # ' + str(new_leg))
    U,S,V = trun(U,S,V,cutoff)
    leg_list[ind] = top_legs + [new_leg]
    tensor_list[ind] = U.reshape(tuple(top_shape + [len(S)]))
    leg_list.append([new_leg] +  bot_legs)
    tensor_list.append(np.dot(np.diag(S),V).reshape(tuple([len(S)] + bot_shape)))
    if len(ent_list) == new_leg:  # append Schmidt values at new_leg index
        ent_list.append(S)
    elif (ent_list[new_leg] == [0])[0]:
        ent_list[new_leg] = S
    return new_leg

############ this function should be generalized to perform a bigger
# matrix multiplication if many legs can be simultaneously contracted

def cont_leg(leg_val, tens_list, leg_list, ent_list):
    """ Contract tensors along leg_val.
    Output updated tensor and leg lists  """
    New_Tens = tens_list
    New_Legs = leg_list
    New_Schm = ent_list
    # search for desired leg value
    o1, o2 = leg_find(leg_val,leg_list)
    # new contracted tensor
    N_T= np.tensordot(New_Tens[o1[0]],New_Tens[o2[0]],[[o1[1]],[o2[1]]])
    # remove contracted legs and update legs data
    New_Legs[o1[0]].remove(leg_val)
    New_Legs[o2[0]].remove(leg_val)
    N_L = New_Legs[o1[0]] + New_Legs[o2[0]]
    m_big   = max(o1[0],o2[0])
    m_small = min(o1[0],o2[0])
    # delete old tens_list/Legs elements
    del New_Tens[m_big]
    del New_Tens[m_small]
    del New_Legs[m_big]
    del New_Legs[m_small]
    # append new tensor with updated leg indicies
    New_Tens.append(N_T)
    New_Legs.append(N_L)
    New_Schm[leg_val] = np.array([0])
    return New_Tens, New_Legs, New_Schm

def mod_mult(qn, R1, T_List, legs_list, S_List):
    print(qn)
    """"contract element # qn (indexed from 0 to 2*l-1) from the top register (R1)
    with the bottom register qudit.
    Then update tensor/legs lists.
    Assuming |+> pre-contracted with CU**2**i in R1."""
    T_List.append(R1[qn])
    out_0 =  - 2 * (qn + 1)
    legs_list.append([0,-qn-1,out_0])
    cont_leg(0,T_List,legs_list,S_List)   # contract bottom qudit intro modular multiplication
    R2_ten_index = [legs.__contains__(out_0) for legs in legs_list].index(True,0)
    R2_leg_index = legs_list[R2_ten_index].index(out_0)
    legs_list[R2_ten_index][R2_leg_index] = 0

################################
######## MPS MEASUREMENT #######
################################

def Measure(G,L):
    NS = len(G)
    C = [0] * NS
    D = [0] * NS
    # Conventional MPS notation -- see Biamonte Lecture Notes 1 -- Figs 1,2
    # Standard MPS i.e. Schmidt values absorbed into A's
    for inc in range(NS-1):
        C[inc] = np.tensordot(G[inc],np.diag(L[inc]),[[2],[0]])
    C[len(G)-1] = G[len(G)-1]  # last site does not have schmidt values above it
    return C

def meas_prob(measurement,Cs):
    n_MM = len(Cs) # numbert of qubits
    # list of measurement outcomes
    bin_rep = [int(x) for x in bin(measurement)[2:]]
    outcomes = [0] * (n_MM - len(bin_rep)) +  bin_rep

    # initialize effect/measurement array
    D = [0] * n_MM
    for inc in range(n_MM):
        if outcomes[inc]==0:
            D[inc] = np.tensordot(Cs[inc],psi0,[[1],[0]])
        if outcomes[inc]==1:
            D[inc] = np.tensordot(Cs[inc],psi1,[[1],[0]])

    # contract MPS ladder
    prob = D[0]
    for q in range(n_MM-1):
        prob = np.tensordot(prob,D[q+1],[[1],[0]])

    return prob[0,0]

#############################
######## TREE NETWORK #######
#############################

def hierarchy_pos(G, root, width=1.5, vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
    """
    If there is a cycle that is reachable from root,
    then result will not be a hierarchy.

    Parameters
    ----------
    G: networkx graph object
    root: the root node of current tree
    width: horizontal space allocated for this branch
            - avoids overlap with other branches
    vert_gap: gap between levels of hierarchy
    vert_loc: vertical location of root
    xcenter: horizontal location of root

    Returns
    -------
    pos: positions of all G.nodes()

    """
    def h_recur(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5,
                  pos = None, parent = None, parsed = []):
        if(root not in parsed):
            parsed.append(root)
            if pos == None:
                pos = {root:(xcenter,vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            neighbors = G.neighbors(root)
            if parent != None:
                neighbors.remove(parent)
            if len(neighbors)!=0:
                dx = width/len(neighbors)
                nextx = xcenter - width/2 - dx/2
                for neighbor in neighbors:
                    nextx += dx
                    pos = h_recur(G,neighbor, width = dx, vert_gap = vert_gap,
                                        vert_loc = vert_loc-vert_gap,
                                        xcenter=nextx, pos=pos,
                                        parent = root, parsed = parsed)
        return pos
    return h_recur(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5)

def mk_tree(leg_list, ent_list, l, root = 1):
    """ Define networkx graphical representation of tree

    Parameters
    ----------
    leg_list: list of tensors legs
    ent_list: Schmidt coefficients stored in legs
    l: binary lenght of N
    root: manual tree root

    Returns
    -------
    grph: networkx graphica object
    pos: node positions
    edge_labels: dictionary asigning labels to graph edges
    weights: edge bond dimension
    colors: edge characters, i.e. virtual/real
    """
    flat_legs = list(itertools.chain.from_iterable(leg_list)) #flat leg list
    min_fl = min(flat_legs)
    max_fl = max(flat_legs)
    # list of edges with terminating tensors sublist ~ dual to legs_list
    leg_edges = [[i,[leg_list.index(legs) for legs in leg_list
                 if legs.__contains__(i)]] for i in range(min_fl,max_fl+1)]

    # define and populate tree graph
    grph = nx.Graph()
    for legs in leg_edges:
        if len(legs[1]) == 1:
            # add physical qudit/qubit edges
            if legs[0] == 0:
                grph.add_edge(max_fl*2,legs[1][0], number = legs[0],
                weight = 2**l, character = 'qudit')
            else:
                grph.add_edge(legs[0],legs[1][0], number = legs[0], weight = 1,
                character = 'phys')
        elif len(legs[1]) == 0:
            print('Don\'t add edge if no tensor exits')
        else:
            # add virtual edge
            grph.add_edge(legs[1][0],legs[1][1], number = legs[0],
                          weight = len(ent_list[legs[0]]), character = 'virt')
    edge_labels = dict([((u,v,),d['number'])
                     for u,v,d in grph.edges(data=True)])
    edges = grph.edges()
    weights = [grph[u][v]['weight'] for u,v in edges]
    colors = [grph[u][v]['character'] for u,v in edges]

    # bottom qudit is the grph root if it exsits
    if 0 in flat_legs:
        pos = hierarchy_pos(grph,max_fl*2)  # extenal qudit node/tensor
    else:
        pos = hierarchy_pos(grph,root)   # whatever is at the top
    return grph, pos, edge_labels, weights, colors

def draw_tree(grph, pos, ent_List):
    """
    Parameters
    ----------
    grph: networkx graphica object
    pos: node positions

    Returns
    -------
    matplotlib plot of tree graph

    """
    # Virtual nodes/edges
    vnodes = [(u) for (u,v) in grph.nodes(data = True)
              if u >= 0 and u < max(grph.nodes())]

    vedges = [(u,v,d['weight']) for (u,v,d) in grph.edges(data = True)
              if d['character'] == 'virt']

    # virtual bond labels dictionary with (t_list index and or dimensionality)
    vrt_labels = dict([((u,v,),(d['number'], len(ent_List[d['number']])))
    # vrt_labels = dict([((u,v,),len(ent_List[d['number']]))
                 for (u,v,d) in grph.edges(data = True)
                 if d['character'] == 'virt'])

    tree_vedges = list(vedges)

    v_width = [np.log2(v[2]) + 1 for v in tree_vedges] # virtual bond dimensions

    # Physical nodes/edges
    pedges = [(u,v,d['weight']) for (u,v,d) in grph.edges(data = True)
              if d['character'] == 'phys']

    p_width = [np.log2(p[2] + 1 ) for p in pedges]

    qb_labels = dict([((u,v,), - d['number']) for (u,v,d)   # label qubits
                    in grph.edges(data = True) if d['character'] == 'phys'])
    # print(vrt_labels, qb_labels)
    # Draw nodes
    nx.draw_networkx_nodes(grph, pos, nodelist = vnodes,
                           node_size = 100, node_color = 'r')
    # with their labels
    # nd_labels = {}
    # for (u,v) in grph.nodes(data = True):
    #     nd_labels[u] = str(u)
    # nx.draw_networkx_labels(grph, pos, nd_labels, font_size=16)

    # Draw edges
    nx.draw_networkx_edges(grph, pos, edgelist = tree_vedges,
                           width = v_width, alpha=1.0,
                           edge_color='k', style='solid')
    vtlabs = nx.draw_networkx_edge_labels(grph, pos, edgelist = tree_vedges,
                                edge_labels = vrt_labels)

    # if graph contains qudit:
    root_node = [(u,v,d['weight']) for (u,v,d)
                 in grph.edges(data = True) if d['character'] == 'qudit']
    # print(pos)
    if root_node != []:
        nx.draw_networkx_edges(grph, pos,
        edgelist = [(root_node[0][0],root_node[0][1],root_node[0][2])],
        width = np.log2(root_node[0][2]) + 1, alpha=1.0, edge_color='r',
        style = 'solid')

    nx.draw_networkx_edges(grph, pos, edgelist = pedges,
                           width = p_width, alpha=1.0,
                           edge_color='r', style='solid')
    qblabs = nx.draw_networkx_edge_labels(grph, pos, edgelist = pedges,
                                 edge_labels = qb_labels)
    # rotate labels horizontally
    for _,t in qblabs.items():
        t.set_rotation('horizontal')
    for _,t in vtlabs.items():
        t.set_rotation('horizontal')

    plt.show()


###############################
########### SHOR'S ############
######### ALGORITHM  ##########
########### GATES  ############
###############################

def U_QFT(n):
    """ Matrix representation of n-qubit QFT """
    dim = 2**n # Hilbert space dimensionality
    Gate= [[np.exp(2 * np.pi * 1j * x * y / dim) for x in range(dim)] for y in range(dim)]
    Gate = np.array(Gate)/np.sqrt(dim)
    return Gate

def MM(x,N,n,l,t=0):
    """Outputs Modular Multiplication (MM) operator U(a)**n performing
    U**n |a> = |a x**n Mod N>
    x: Random integer 1<x<N
    N: Modular parameter
    n: Exponent for U
    l: Number of qubits in bottomr register
    """
    Mat = np.zeros([2**l,2**l])
    for iii in range(N):
        Mat[iii,(x**n * iii)%N] = 1
    return Mat

def TMM(x,N,n,trun_basis):
    """
    Truncated modular multiplication gate.
    U**n |a> = |a x**n Mod N>  ---  only involving included basis states |a>!!!
    x: Random integer 1<x<N
    N: Modular parameter
    n: Exponent for U
    trun_basis: truncated basis for bottom register
    """
    Mat = np.zeros([len(trun_basis),len(trun_basis)])
    print('making TMM')
    perms = [int((x**n * iii)%N) for iii in trun_basis] # Modular multiplication
    for iii in range(len(trun_basis)):
        if trun_basis.__contains__(perms[iii]):
            Mat[iii,trun_basis.index(perms[iii])] = 1
    return Mat

def MPS_CRS(Ab, S, At, s):
    """
    MPS friendly QFT circuit with properties:
    1) it is implemented in a recursive block-wise manner
    2) only 2-local gates gates are performed
    -  controlled rotation + SWAP (CRS) on neighboring qubits
    3) the input and output of the QFT function are MPS
    Inputs are:
    At,Ab: top/bottom tensors
    S: Schmidt coeffs
    s: site of target. I.e. CRS from s(target) to s+1(control)
    """
    # contract two on-site tensors into 2-site tensor
    T_bot = np.tensordot(Ab,np.diag(S),[[2],[0]])
    pre_shape = T_bot.shape[:len(T_bot.shape)-2] + (4,) + At.shape[2:]   # shape of 2 sites tensor before CRS operator
    post_shape = T_bot.shape[:len(T_bot.shape)-1] + At.shape[1:]         # shape after CRS application, but before SVD
    T_2_psi = (np.tensordot(T_bot,A[s+1],[[2],[0]])).reshape(pre_shape)

    # apply rotation and swap operators, then swapaxes to bring back to MPS form, reshape and swap qubit axes as needed
    New_T_2 = np.tensordot(T_2_psi,CRS(s+1),[[1],[0]])
    New_T_2 = (New_T_2.swapaxes(1,2)).reshape(post_shape)
    New_T_2 = New_T_2#.swapaxes(1,2)                      # !!!!VERIFY THIS STEP!!!! # consequence of our qubits being numbered from bottom to top

    # group indicies on each side of cut and perform SVD, then call truncation function
    U, S, V = np.linalg.svd(New_T_2.reshape(np.prod(New_T_2.shape[0:2]),np.prod(New_T_2.shape[2:4])))
    return trun(U,S,V,0.0001)
