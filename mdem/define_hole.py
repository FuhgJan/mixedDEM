import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from utilities.delaunay_integration import delaunayInt_Numpy, Delaunay_Constraint
from utilities.utility import getEquidistantPoints,sampleEdges,getDistance
import torch
from pyevtk.hl import gridToVTK
import random as rnd
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle, VtkQuad

def checkIfInCircle(x,r,c):
    distances = getDistance(x, c)
    idx = np.where(distances>(r+0.005))[0]
    return x[idx,:]


def sampleCircle(r,c,n):
    theta = np.linspace(0,2*np.pi,n)
    x = c[0] + r*np.cos(theta)
    y = c[1] + r*np.sin(theta)

    p = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)),axis=1)

    normals = np.zeros((n,2))
    for i in range(n):
        n = c-p[i,:]
        n = n/np.linalg.norm(n)

        normals[i,:] = n

    return p,normals


def getDistancesAndValues(x):
    n = 5000

    interface_u_BC = np.array([[0.0, 0.0], [0.0, 2.0]])
    interface_u_edge = np.array([[0, 1]])
    bcuPoints1 = sampleEdges(interface_u_BC,interface_u_edge,n)

    interface_P11_BC = np.array([[2.0, 0.0], [2.0, 1.0]])
    bcP11Points = sampleEdges(interface_P11_BC,interface_u_edge,n)
    interface_P21_BC = np.array([[1.0, 0.0], [2.0, 1.0]])
    bcP21Points = sampleEdges(interface_P21_BC,interface_u_edge,1000)

    interface_P12_BC = np.array([ [0.0, 1.0], [2.0, 1.0]])
    interface_P12_edge = np.array([[0, 1]])
    bcP12Points = sampleEdges(interface_P12_BC,interface_P12_edge,n)

    interface_P22_BC = np.array([ [0.0, 1.0], [2.0, 1.0]])
    interface_P22_edge = np.array([[0, 1]])
    bcP22Points = sampleEdges(interface_P22_BC,interface_P22_edge,n)


    d_u1 = np.zeros((x.shape[0], 1))
    d_u2 = np.zeros((x.shape[0], 1))

    d_P11 = np.zeros((x.shape[0], 1))
    d_P12 = np.zeros((x.shape[0], 1))
    d_P21 = np.zeros((x.shape[0], 1))
    d_P22 = np.zeros((x.shape[0], 1))


    val_u1 = np.zeros((bcuPoints1.shape[0],6))
    val_u2 = np.zeros((bcuPoints1.shape[0],6))

    val_P22 = np.zeros((bcP22Points.shape[0],6))

    val_P12 = np.zeros((bcP12Points.shape[0],6))
    val_P21 = np.zeros((bcP21Points.shape[0],6))

    idxG = np.where(((bcP21Points[:,0]>=0.0) & (bcP21Points[:,0]<=2.0) & (bcP21Points[:,1]==1.0)))[0]
    val_P21[idxG,4] = -100.0*bcP21Points[idxG,0]

    val_P11 = np.zeros((bcP11Points.shape[0],6))

    inputValues = np.concatenate((bcuPoints1, bcuPoints1,  bcP11Points, bcP12Points, bcP21Points, bcP22Points ), axis=0)
    outputValues = np.concatenate((val_u1, val_u2,  val_P11, val_P12, val_P21, val_P22 ), axis=0)


    areaOut = np.zeros((x.shape[0],6))


    inputValues = np.concatenate((inputValues, x), axis=0)
    outputValues =np.concatenate((outputValues, areaOut), axis=0)


    idxIn = np.where((inputValues[:,1]==1.0))[0]
    outputValues[idxIn,4] = -100*inputValues[idxIn,0]

    idxIn = np.where((inputValues[:,1]==1.0))[0]

    for i in range(x.shape[0]):
        d_u1[i] = np.min(getDistance(bcuPoints1, x[i,:]))
        d_u2[i] = np.min(getDistance(bcuPoints1, x[i,:]))

        d_P11[i] = np.min(getDistance(bcP11Points, x[i, :]))
        d_P12[i] = np.min(getDistance(bcP12Points, x[i, :]))
        d_P21[i] = np.min(getDistance(bcP21Points, x[i, :]))
        d_P22[i] = np.min(getDistance(bcP22Points, x[i, :]))


    distances =  np.concatenate((d_u1, d_u2,  d_P11, d_P12, d_P21, d_P22 ), axis=1)

    bccircle,bcnormals = sampleCircle(0.1, np.array([0.5,0.5]), n)

    return distances, inputValues, outputValues, bccircle,bcnormals



# ----------------------------- define structural parameters ---------------------------------------
Length =2.0
Height = 1.0
Depth = 1.0

# ------------------------------ define domain and collocation points -------------------------------
Nx = 100 # 120  # 120
Ny = 50# 30  # 60
x_min, y_min = (0.0, 0.0)
hx = Length / (Nx - 1)
hy = Height / (Ny - 1)
shape = [Nx, Ny]
dxdy = [hx, hy]

# ------------------------------ data testing -------------------------------------------------------
num_test_x = 75
num_test_y = 75

cords = [(0.0, 0.0), (0.0, 1.0), (2.0, 1.0), (2.0, 0.0), (0.0, 0.0)]


def boundaryf():
    n = 5000
    # Edges P11
    interface_edgePoints =np.array([[2.0, 0.0], [2.0, 1.0]])
    interface_edges = np.array([[0, 1]])
    pointsP11 = sampleEdges(interface_edgePoints, interface_edges, n)
    valP11 = np.zeros((pointsP11.shape[0], 1))



    # Edges P21
    interface_edgePoints = np.array([[2.0, 0.0], [2.0, 1.0], [0.0, 0.0], [0.0, 1.0]])
    interface_edges = np.array([[0, 1],[2,3]])
    pointsP21 = sampleEdges(interface_edgePoints, interface_edges, 2*n)
    valP21 = np.zeros((pointsP21.shape[0], 1))



    # Edges P12
    interface_edgePoints = np.array([[0.0, 1.0], [2.0, 1.0]])
    interface_edges = np.array([[0, 1]])
    pointsP12 = sampleEdges(interface_edgePoints, interface_edges, n)
    valP12 = -10.0 * pointsP12[:, 0:1]


    # Edges P22
    interface_edgePoints =np.array([[0.0, 1.0], [2.0, 1.0]])
    interface_edges = np.array([[0, 1]])
    pointsP22 = sampleEdges(interface_edgePoints, interface_edges, n)
    valP22 =  np.zeros((pointsP22.shape[0], 1)) #-10.0*pointsP22[:,0:1]


    return pointsP11,valP11,  pointsP21,valP21, pointsP12,valP12, pointsP22,valP22







def setup_domain():
    x_dom = x_min, Length, Nx
    y_dom = y_min, Height, Ny
    # create points
    lin_x = np.linspace(0, 2, x_dom[2])
    lin_y = np.linspace(0, 1, y_dom[2])
    dom = np.zeros((Nx * Ny, 2))
    c = 0
    for x in np.nditer(lin_x):
        tb = y_dom[2] * c
        te = tb + y_dom[2]
        c += 1
        dom[tb:te, 0] = x
        dom[tb:te, 1] = lin_y

    Nx1 = 50
    Ny1 = 50
    lin_x = np.linspace(0.25, 0.75, Nx1)
    lin_y = np.linspace(0.25, 0.75, Ny1)
    dom_finer = np.zeros((Nx1 * Ny1, 2))
    c = 0
    for x in np.nditer(lin_x):
        tb =Nx1 * c
        te = tb + Ny1
        c += 1
        dom_finer[tb:te, 0] = x
        dom_finer[tb:te, 1] = lin_y

    Nx2 = 30
    Ny2 = 30
    lin_x = np.linspace(1.9, 2.0, Nx2)
    lin_y = np.linspace(0, 1, Ny2)
    dom_right = np.zeros((Nx2 * Ny2, 2))
    c = 0
    for x in np.nditer(lin_x):
        tb = Nx2 * c
        te = tb + Ny2
        c += 1
        dom_right[tb:te, 0] = x
        dom_right[tb:te, 1] = lin_y


    dom = np.concatenate((dom_finer, dom, dom_right))
    dom = checkIfInCircle(dom,0.1,np.array([0.5,0.5]))
    n = 50
    circle,normals = sampleCircle(0.1, np.array([0.5,0.5]), n)
    segments_circle = np.zeros((n,2))
    for i in range(n):
        segments_circle[i,0]=i
        if i==n-1:
            segments_circle[i, 1] = 0
        else:
            segments_circle[i,1]= i+1



    dom_new = np.concatenate((circle, dom))
    distances, inputValues, outputValues, bccircle,bcnormals = getDistancesAndValues(dom_new)

    edge0 = np.where(((np.abs(dom_new[:, 1] - cords[0][1])) < 1e-3) & ((np.abs(dom_new[:, 0] - cords[0][0])) < 1e-3))[0][0]
    edge1 = np.where(((np.abs(dom_new[:, 1] - cords[1][1])) < 1e-3) & ((np.abs(dom_new[:, 0] - cords[1][0])) < 1e-3))[0][0]
    edge2 = np.where(((np.abs(dom_new[:, 1] - cords[2][1])) < 1e-3) & ((np.abs(dom_new[:, 0] - cords[2][0])) < 1e-3))[0][0]
    edge3 = np.where(((np.abs(dom_new[:, 1] - cords[3][1])) < 1e-3) & ((np.abs(dom_new[:, 0] - cords[3][0])) < 1e-3))[0][0]


    segments_norm = np.array(
        [[edge0, edge1 ] ,[edge1,  edge2] , [edge2, edge3], [edge3, edge0]])

    segments = np.concatenate((segments_circle, segments_norm))
    pointsSection_torch = torch.from_numpy(dom_new)
    segments_torch = torch.from_numpy(segments)
    holes = np.array([[0.5, 0.5]])
    holes_torch = torch.from_numpy(holes)

    delaun = Delaunay_Constraint(pointsSection_torch,segments_torch,holes=holes_torch, val = 'p')


    interface_edgePoints = np.array([[0.0, 0.0], [0.0, 1.0]])
    interface_edges = np.array([[0, 1]])
    bcl_u_pts_left = sampleEdges(interface_edgePoints,interface_edges,1000)
    bcl_u_left = np.ones(np.shape(bcl_u_pts_left))
    bcl_u_left[:,0] = 0

    interface_edgePoints = np.array([[0.0, 0.0], [2.0, 0.0]])
    interface_edges = np.array([[0, 1]])
    bcl_u_pts_bottom = sampleEdges(interface_edgePoints,interface_edges,1000)
    bcl_u_bottom= np.zeros(np.shape(bcl_u_pts_bottom))


    interface_edgePoints = np.array([[0.0, 1.0], [2.0, 1.0]])
    interface_edges = np.array([[0, 1]])

    interfacePoints = sampleEdges(interface_edgePoints,interface_edges,1000)

    bcr_t_pts = interfacePoints
    bcr_t = interfacePoints[:,0:1] * [-10.0, 0.0]


    pointsP11, valP11, pointsP21, valP21, pointsP12, valP12, pointsP22, valP22 = boundaryf()


    boundary = {
        'coordsP11': pointsP11,
        'valP11': valP11,

        'coordsP21': pointsP21,
        'valP21': valP21,

        'coordsP12': pointsP12,
        'valP12': valP12,

        'coordsP22': pointsP22,
        'valP22': valP22,
    }


    delaunayIntegration = {
        # condition on the right
        "del1": delaun,
        # adding more boundary condition here ...
    }

    tractionFreeCircle = {
        # condition on the right
        'coord':bccircle,
        'norm':bcnormals,
        # adding more boundary condition here ...
    }


    boundary_neumann = {
        # condition on the right
        "neumann_1": {
            "coord": bcr_t_pts,
            "known_value": bcr_t,
            "bcpPoints": boundary
        }
        # adding more boundary condition here ...
    }
    boundary_dirichlet = {
        # condition on the left
        "dirichlet_1": {
            "coord_left": bcl_u_pts_left,
            "coord_bottom": bcl_u_pts_bottom,
            "known_value_left": bcl_u_left,
            "known_value_bottom": bcl_u_bottom,

        }
        # adding more boundary condition here ...
    }
    return dom_new, boundary_neumann, boundary_dirichlet,delaunayIntegration, tractionFreeCircle, distances, inputValues, outputValues



def get_datatest(Nx=num_test_x, Ny=num_test_y):
    lin_x = np.linspace(0, 2, Nx)
    lin_y = np.linspace(0, 1, Ny)
    dom = np.zeros((Nx * Ny, 2))
    c = 0
    for x in np.nditer(lin_x):
        tb = Ny * c
        te = tb + Ny
        c += 1
        dom[tb:te, 0] = x
        dom[tb:te, 1] = lin_y


    n = 35
    circle,normals = sampleCircle(0.1, np.array([0.5,0.5]), n)
    segments_circle = np.zeros((n,2))
    for i in range(n):
        segments_circle[i,0]=i
        if i==n-1:
            segments_circle[i, 1] = 0
        else:
            segments_circle[i,1]= i+1

    dom = checkIfInCircle(dom,0.1,np.array([0.5,0.5]))

    dom = np.concatenate((circle, dom))

    return dom



def save_WithHole(dom, filename_out, U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23,
                                  E33, SVonMises,
                                  F11, F12, F21, F22, P11, P12, P13, P21, P22, P23, P31, P32, P33):
    domCom = get_datatest(num_test_x, num_test_y)
    dom = domCom

    n = 35
    segments_circle = np.zeros((n,2))
    for i in range(n):
        segments_circle[i,0]=i
        if i==n-1:
            segments_circle[i, 1] = 0
        else:
            segments_circle[i,1]= i+1

    edge0 = np.where(((np.abs(dom[:, 1] - cords[0][1])) < 1e-3) & ((np.abs(dom[:, 0] - cords[0][0])) < 1e-3))[0][0]
    edge1 = np.where(((np.abs(dom[:, 1] - cords[1][1])) < 1e-3) & ((np.abs(dom[:, 0] - cords[1][0])) < 1e-3))[0][0]
    edge2 = np.where(((np.abs(dom[:, 1] - cords[2][1])) < 1e-3) & ((np.abs(dom[:, 0] - cords[2][0])) < 1e-3))[0][0]
    edge3 = np.where(((np.abs(dom[:, 1] - cords[3][1])) < 1e-3) & ((np.abs(dom[:, 0] - cords[3][0])) < 1e-3))[0][0]


    segments_norm = np.array(
        [[edge0, edge1 ] ,[edge1,  edge2] , [edge2, edge3], [edge3, edge0]])

    segments = np.concatenate((segments_circle, segments_norm))
    pointsSection_torch = torch.from_numpy(dom)
    segments_torch = torch.from_numpy(segments)
    holes = np.array([[0.5, 0.5]])
    holes_torch = torch.from_numpy(holes)


    delaun = Delaunay_Constraint(pointsSection_torch,segments_torch,holes=holes_torch, val = 'p')


    tets = delaun.d['triangles']

    tetsLast = tets[:,2]
    idxSort = np.argsort(tetsLast)
    tetsNew = tets[idxSort,:]


    conn = tetsNew.flatten()


    n = tetsNew.shape[0]
    ctype = np.zeros(n)
    off= np.zeros(n)
    for i in range(n):
        ctype[i]=VtkTriangle.tid
        if i==0:
            off[i] = 3
        else:
            off[i] = off[i-1]+3
    x = delaun.d['vertices'][:,0]
    y = delaun.d['vertices'][:, 1]
    z = np.zeros( dom[:, 1].shape)

    conn = np.ascontiguousarray(conn, dtype=np.int)
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
    z = np.ascontiguousarray(z, dtype=np.float32)
    U = np.ascontiguousarray(U, dtype=np.float32)
    SVonMises = np.ascontiguousarray(SVonMises, dtype=np.float32)
    S11 = np.ascontiguousarray(S11, dtype=np.float32)
    S12 = np.ascontiguousarray(S12, dtype=np.float32)
    S13 = np.ascontiguousarray(S13, dtype=np.float32)
    S22 = np.ascontiguousarray(S22, dtype=np.float32)
    S23 = np.ascontiguousarray(S23, dtype=np.float32)
    S33 = np.ascontiguousarray(S33, dtype=np.float32)
    E11 = np.ascontiguousarray(E11, dtype=np.float32)
    E12 = np.ascontiguousarray(E12, dtype=np.float32)
    E13 = np.ascontiguousarray(E13, dtype=np.float32)
    E22 = np.ascontiguousarray(E22, dtype=np.float32)
    E23 = np.ascontiguousarray(E23, dtype=np.float32)
    E33 = np.ascontiguousarray(E33, dtype=np.float32)

    F11 = np.ascontiguousarray(F11, dtype=np.float32)
    F12 = np.ascontiguousarray(F12, dtype=np.float32)
    F21 = np.ascontiguousarray(F21, dtype=np.float32)
    F22 = np.ascontiguousarray(F22, dtype=np.float32)
    P11 = np.ascontiguousarray(P11, dtype=np.float32)
    P12 = np.ascontiguousarray(P12, dtype=np.float32)
    P13 = np.ascontiguousarray(P13, dtype=np.float32)
    P21 = np.ascontiguousarray(P21, dtype=np.float32)
    P22 = np.ascontiguousarray(P22, dtype=np.float32)
    P23 = np.ascontiguousarray(P23, dtype=np.float32)
    P31 = np.ascontiguousarray(P31, dtype=np.float32)
    P32 = np.ascontiguousarray(P32, dtype=np.float32)
    P33 = np.ascontiguousarray(P33, dtype=np.float32)

    off = np.ascontiguousarray(off, dtype=np.int)

    point_data ={"u1": U[0,:],"u2": U[1,:],"u3": U[2,:], "S-VonMises": SVonMises[:,0], \
                                               "S11": S11[:,0], "S12": S12[:,0], "S13": S13[:,0], \
                                               "S22": S22[:,0], "S23": S23[:,0], "S33": S33[:,0], \
                                               "E11": E11[:,0], "E12": E12[:,0], "E13": E13[:,0], \
                                               "E22": E22[:,0], "E23": E23[:,0], "E33": E33[:,0], \
                                               "F11": F11[:,0], "F12": F12[:,0], "F21": F21[:,0], \
                                               "F22": F22[:,0], 'P11':P11[:,0], 'P12':P12[:,0],'P13':P13[:,0],'P21':P21[:,0], 'P22':P22[:,0],'P23':P23[:,0], 'P31':P31[:,0],'P32':P32[:,0],'P33':P33[:,0]}

    unstructuredGridToVTK(
        filename_out,
        x,
        y,
        z,
    connectivity=conn,
    cell_types=ctype,
        offsets=off,
        pointData=point_data,

    )

    return dom



if __name__ == '__main__':
    #get_datatest(50,50)
    setup_domain()
    #print('Dataset')
    #dom = get_datatest(num_test_x, num_test_y)
    #save_WithHole(dom,'Test',None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None)
