import torch
from torch.autograd import grad
import numpy as np
import numpy.random as npr
from matplotlib import cm
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from pyevtk.hl import gridToVTK
from pyevtk.hl import pointsToVTK
import numpy.matlib as ml

from scipy import interpolate


dev = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    dev = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("CUDA not available, running on CPU")


def getDistance(x,y):
    return np.sqrt(np.power(x[:,0]-y[0],2) + np.power(x[:,1]-y[1],2))

def plotContour(x,f, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if torch.is_tensor(x):

        X, Y = np.mgrid[x[:, 0].detach().numpy().min():x[:, 0].detach().numpy().max():50j,
               x[:, 1].detach().numpy().min():x[:, 1].detach().numpy().max():50j]
        points = np.c_[X.ravel(), Y.ravel()]
        V = interpolate.griddata(np.c_[x[:, 0].detach().numpy(), x[:, 1].detach().numpy()], f.detach().numpy(),
                                 points).reshape(X.shape)
    else:
        X, Y = np.mgrid[x[:, 0].min():x[:, 0].max():50j,
               x[:, 1].min():x[:, 1].max():50j]
        points = np.c_[X.ravel(), Y.ravel()]
        V = interpolate.griddata(np.c_[x[:, 0], x[:, 1]], f,
                                 points).reshape(X.shape)
    # cs = ax.pcolor(zz[:,:,0])
    cs = ax.pcolor(X, Y, V)
    # plt.contourf(xx,yy,zz) # if you want contour plot
    # plt.imshow(zz)
    # plt.pcolorbar()
    cbar = plt.colorbar(cs, ax=ax)
    cbar.solids.set_edgecolor("face")
    if filename==None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close(fig)



def getEquidistantPoints(p1, p2, parts):
    return zip(np.linspace(p1[0], p2[0], parts+1),
               np.linspace(p1[1], p2[1], parts+1))

def sampleEdges(x,edges, nPointsPerEdge):
    r = []
    for e in edges:
        x1 = x[e[0], :]
        x2 = x[e[1], :]
        out = np.asarray(list(getEquidistantPoints(x1, x2, nPointsPerEdge)))
        r.append(out)



    r1 = np.asarray(r)
    r2 = r1.reshape((r1.shape[0] * r1.shape[1], 2))

    return r2



# convert numpy BCs to torch
def ConvBCsToTensors(bc_d):
    size_in_1 = len(bc_d)
    size_in_2 = len(bc_d[0][0])
    bc_in = torch.empty(size_in_1, size_in_2, device=dev)
    c = 0
    for bc in bc_d:
        bc_in[c, :] = torch.from_numpy(bc[0])
        c += 1
    return bc_in


# --------------------------------------------------------------------------------
# purpose: doing something in post processing for visualization in 3D
# --------------------------------------------------------------------------------
def write_vtk(filename, x_space, y_space, z_space, Ux, Uy, Uz):
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    displacement = (Ux, Uy, Uz)
    gridToVTK(filename, xx, yy, zz, pointData={"displacement": displacement})

# --------------------------------------------------------------------------------
# purpose: doing something in post processing for visualization in 3D
# --------------------------------------------------------------------------------
def write_vtk_v2(filename, x_space, y_space, z_space, U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises):
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    gridToVTK(filename, xx, yy, zz, pointData={"displacement": U, "S-VonMises": SVonMises, \
                                               "S11": S11, "S12": S12, "S13": S13, \
                                               "S22": S22, "S23": S23, "S33": S33, \
                                               "E11": E11, "E12": E12, "E13": E13, \
                                               "E22": E22, "E23": E23, "E33": E33\
                                               })
    # gridToVTK(filename, xx, yy, zz, pointData={"displacement": U})


def write_vtk_6out(filename, x_space, y_space, z_space, U,P11, P12,P13,P21, P22,P23, P31,P32,P33):
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    gridToVTK(filename, xx, yy, zz, pointData={"displacement": U,'P11':P11, 'P12':P12,'P13':P13,'P21':P21, 'P22':P22,'P23':P23, 'P31':P31,'P32':P32,'P33':P33                                           \
                                                })


def write_vtk_v3(filename, x_space, y_space, z_space, U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, F11, F12, F21, F22):
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    gridToVTK(filename, xx, yy, zz, pointData={"displacement": U, "S-VonMises": SVonMises, \
                                               "S11": S11, "S12": S12, "S13": S13, \
                                               "S22": S22, "S23": S23, "S33": S33, \
                                               "E11": E11, "E12": E12, "E13": E13, \
                                               "E22": E22, "E23": E23, "E33": E33, \
                                               "F11": F11, "F12": F12, "F21": F21, \
                                               "F22": F22                                           \
                                                })



    # gridToVTK(filename, xx, yy, zz, pointData={"displacement": U})


def write_vtk_v4(filename, x_space, y_space, z_space, U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, F11, F12, F21, F22,P11, P12,P13,P21, P22,P23, P31,P32,P33):
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    gridToVTK(filename, xx, yy, zz, pointData={"displacement": U, "S-VonMises": SVonMises, \
                                               "S11": S11, "S12": S12, "S13": S13, \
                                               "S22": S22, "S23": S23, "S33": S33, \
                                               "E11": E11, "E12": E12, "E13": E13, \
                                               "E22": E22, "E23": E23, "E33": E33, \
                                               "F11": F11, "F12": F12, "F21": F21, \
                                               "F22": F22, 'P11':P11, 'P12':P12,'P13':P13,'P21':P21, 'P22':P22,'P23':P23, 'P31':P31,'P32':P32,'P33':P33                                           \
                                                })


def write_vtk_v5(filename, x_space, y_space, z_space, U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, F11, F12, F21, F22,P11, P12,P13,P21, P22,P23, P31,P32,P33, out1, out2):
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    gridToVTK(filename, xx, yy, zz, pointData={"displacement": U, "S-VonMises": SVonMises, \
                                               "S11": S11, "S12": S12, "S13": S13, \
                                               "S22": S22, "S23": S23, "S33": S33, \
                                               "E11": E11, "E12": E12, "E13": E13, \
                                               "E22": E22, "E23": E23, "E33": E33, \
                                               "F11": F11, "F12": F12, "F21": F21, \
                                               "F22": F22, 'P11':P11, 'P12':P12,'P13':P13,'P21':P21, 'P22':P22,'P23':P23, 'P31':P31,'P32':P32,'P33':P33,'out1':out1,'out2':out2                                           \
                                                })


def write_Single(filename, x_space, y_space, z_space, A):
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    gridToVTK(filename, xx, yy, zz, pointData={"A": A
                                               })


def write_vtk_v2_linElasticity(filename, x_space, y_space, z_space, U, S11, S12, S13, S22, S23, S33, SVonMises, eps11, eps12, eps13, eps22, eps23, eps33):
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    gridToVTK(filename, xx, yy, zz, pointData={"displacement": U, "S-VonMises": SVonMises, \
                                               "S11": S11, "S12": S12, "S13": S13, \
                                               "S22": S22, "S23": S23, "S33": S33,
                                               "eps11": eps11, "eps12": eps12, "eps13": eps13, \
                                               "eps22": eps22, "eps23": eps23, "eps33": eps33,
                                               })
    # gridToVTK(filename, xx, yy, zz, pointData={"displacement": U})


def write_points_toVtk(filename, x, y, z, U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises):
    pointsToVTK(filename, x, y, z, data={"displacement": U, "S-VonMises": SVonMises, \
                                               "S11": S11, "S12": S12, "S13": S13, \
                                               "S22": S22, "S23": S23, "S33": S33, \
                                               "E11": E11, "E12": E12, "E13": E13, \
                                               "E22": E22, "E23": E23, "E33": E33\
                                               })

def write_points_toVtk_LinElast(filename, x, y, z, U, S11, S12, S13, S22, S23, S33, SVonMises):
    pointsToVTK(filename, x, y, z, data={"displacement": U, "S-VonMises": SVonMises, \
                                               "S11": S11, "S12": S12, "S13": S13, \
                                               "S22": S22, "S23": S23, "S33": S33
                                               })


# --------------------------------------------------------------------------------
# purpose: doing something in post processing for visualization in 3D
# --------------------------------------------------------------------------------
def write_arr2DVTK(filename, coordinates, values):
    # displacement = np.concatenate((values[:, 0:1], values[:, 1:2], values[:, 0:1]), axis=1)
    x = np.array(coordinates[:, 0].flatten(), dtype='float32')
    y = np.array(coordinates[:, 1].flatten(), dtype='float32')
    z = np.zeros(x.shape, dtype='float32')
    disX = np.array(values[:, 0].flatten(), dtype='float32')
    disY = np.array(values[:, 1].flatten(), dtype='float32')
    disZ = np.zeros(disX.shape, dtype='float32')
    displacement = (disX, disY, disZ)
    gridToVTK(filename, x, y, z, pointData={"displacement": displacement})

# --------------------------------------------------------------------------------
# purpose: doing something in post processing for visualization in 3D
# --------------------------------------------------------------------------------
def write_vtk_2d(filename, x_space, y_space, Ux, Uy):
    xx, yy = np.meshgrid(x_space, y_space)
    displacement = (Ux, Uy, Ux)
    gridToVTK(filename, xx, yy, xx,  pointData={"displacement": displacement})




