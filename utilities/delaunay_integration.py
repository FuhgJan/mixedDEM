import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import torch
from shapely.geometry import Point, MultiPoint, Polygon
import numpy as np
from math import sqrt
import triangle as tr

dev = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    dev = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("CUDA not available, running on CPU")



class Delaunay2D_Numpy:
    """
    Class to compute a Delaunay triangulation in 2D
    ref: http://en.wikipedia.org/wiki/Bowyer-Watson_algorithm
    ref: http://www.geom.uiuc.edu/~samuelp/del_project.html
    """

    def __init__(self, center=(0, 0), radius=9999):
        """ Init and create a new frame to contain the triangulation
        center -- Optional position for the center of the frame. Default (0,0)
        radius -- Optional distance from corners to the center.
        """
        center = np.asarray(center)
        # Create coordinates for the corners of the frame
        self.coords = [center+radius*np.array((-1, -1)),
                       center+radius*np.array((+1, -1)),
                       center+radius*np.array((+1, +1)),
                       center+radius*np.array((-1, +1))]

        # Create two dicts to store triangle neighbours and circumcircles.
        self.triangles = {}
        self.circles = {}

        # Create two CCW triangles for the frame
        T1 = (0, 1, 3)
        T2 = (2, 3, 1)
        self.triangles[T1] = [T2, None, None]
        self.triangles[T2] = [T1, None, None]

        # Compute circumcenters and circumradius for each triangle
        for t in self.triangles:
            self.circles[t] = self.circumcenter(t)

    def circumcenter(self, tri):
        """Compute circumcenter and circumradius of a triangle in 2D.
        Uses an extension of the method described here:
        http://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html
        """
        pts = np.asarray([self.coords[v] for v in tri])
        #pts = pts.astype('float32')
        pts2 = np.dot(pts, pts.T)
        A = np.bmat([[2 * pts2, [[1],
                                 [1],
                                 [1]]],
                      [[[1, 1, 1, 0]]]])
        #A = A.astype('float32')
        b = np.hstack((np.sum(pts * pts, axis=1), [1]))
        #b = b.astype('float32')
        x = np.linalg.solve(A, b)
        bary_coords = x[:-1]
        center = np.dot(bary_coords, pts)

        # radius = np.linalg.norm(pts[0] - center) # euclidean distance
        radius = np.sum(np.square(pts[0] - center))  # squared distance
        return (center, radius)

    def inCircleFast(self, tri, p):
        """Check if point p is inside of precomputed circumcircle of tri.
        """
        center, radius = self.circles[tri]
        return np.sum(np.square(center - p)) <= radius

    def inCircleRobust(self, tri, p):
        """Check if point p is inside of circumcircle around the triangle tri.
        This is a robust predicate, slower than compare distance to centers
        ref: http://www.cs.cmu.edu/~quake/robust.html
        """
        m1 = np.asarray([self.coords[v] - p for v in tri])
        m2 = np.sum(np.square(m1), axis=1).reshape((3, 1))
        m = np.hstack((m1, m2))    # The 3x3 matrix to check
        return np.linalg.det(m) <= 0

    def addPoint(self, p):
        """Add a point to the current DT, and refine it using Bowyer-Watson.
        """
        p = np.asarray(p)
        idx = len(self.coords)
        # print("coords[", idx,"] ->",p)
        self.coords.append(p)

        # Search the triangle(s) whose circumcircle contains p
        bad_triangles = []
        for T in self.triangles:
            # Choose one method: inCircleRobust(T, p) or inCircleFast(T, p)
            if self.inCircleFast(T, p):
                bad_triangles.append(T)

        # Find the CCW boundary (star shape) of the bad triangles,
        # expressed as a list of edges (point pairs) and the opposite
        # triangle to each edge.
        boundary = []
        # Choose a "random" triangle and edge
        T = bad_triangles[0]
        edge = 0
        # get the opposite triangle of this edge
        while True:
            # Check if edge of triangle T is on the boundary...
            # if opposite triangle of this edge is external to the list
            tri_op = self.triangles[T][edge]
            if tri_op not in bad_triangles:
                # Insert edge and external triangle into boundary list
                boundary.append((T[(edge+1) % 3], T[(edge-1) % 3], tri_op))

                # Move to next CCW edge in this triangle
                edge = (edge + 1) % 3

                # Check if boundary is a closed loop
                if boundary[0][0] == boundary[-1][1]:
                    break
            else:
                # Move to next CCW edge in opposite triangle
                edge = (self.triangles[tri_op].index(T) + 1) % 3
                T = tri_op

        # Remove triangles too near of point p of our solution
        for T in bad_triangles:
            del self.triangles[T]
            del self.circles[T]

        # Retriangle the hole left by bad_triangles
        new_triangles = []
        for (e0, e1, tri_op) in boundary:
            # Create a new triangle using point p and edge extremes
            T = (idx, e0, e1)

            # Store circumcenter and circumradius of the triangle
            self.circles[T] = self.circumcenter(T)

            # Set opposite triangle of the edge as neighbour of T
            self.triangles[T] = [tri_op, None, None]

            # Try to set T as neighbour of the opposite triangle
            if tri_op:
                # search the neighbour of tri_op that use edge (e1, e0)
                for i, neigh in enumerate(self.triangles[tri_op]):
                    if neigh:
                        if e1 in neigh and e0 in neigh:
                            # change link to use our new triangle
                            self.triangles[tri_op][i] = T

            # Add triangle to a temporal list
            new_triangles.append(T)

        # Link the new triangles each another
        N = len(new_triangles)
        for i, T in enumerate(new_triangles):
            self.triangles[T][1] = new_triangles[(i+1) % N]   # next
            self.triangles[T][2] = new_triangles[(i-1) % N]   # previous

    def exportTriangles(self):
        """Export the current list of Delaunay triangles
        """
        # Filter out triangles with any vertex in the extended BBox
        return [(a-4, b-4, c-4)
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]

    def exportCircles(self):
        """Export the circumcircles as a list of (center, radius)
        """
        # Remember to compute circumcircles if not done before
        # for t in self.triangles:
        #     self.circles[t] = self.circumcenter(t)

        # Filter out triangles with any vertex in the extended BBox
        # Do sqrt of radius before of return
        return [(self.circles[(a, b, c)][0], sqrt(self.circles[(a, b, c)][1]))
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]

    def exportDT(self):
        """Export the current set of Delaunay coordinates and triangles.
        """
        # Filter out coordinates in the extended BBox
        coord = self.coords[4:]

        # Filter out triangles with any vertex in the extended BBox
        tris = [(a-4, b-4, c-4)
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]
        return coord, tris

    def exportExtendedDT(self):
        """Export the Extended Delaunay Triangulation (with the frame vertex).
        """
        return self.coords, list(self.triangles)

    def exportVoronoiRegions(self):
        """Export coordinates and regions of Voronoi diagram as indexed data.
        """
        # Remember to compute circumcircles if not done before
        # for t in self.triangles:
        #     self.circles[t] = self.circumcenter(t)
        useVertex = {i: [] for i in range(len(self.coords))}
        vor_coors = []
        index = {}
        # Build a list of coordinates and one index per triangle/region
        for tidx, (a, b, c) in enumerate(sorted(self.triangles)):
            vor_coors.append(self.circles[(a, b, c)][0])
            # Insert triangle, rotating it so the key is the "last" vertex
            useVertex[a] += [(b, c, a)]
            useVertex[b] += [(c, a, b)]
            useVertex[c] += [(a, b, c)]
            # Set tidx as the index to use with this triangle
            index[(a, b, c)] = tidx
            index[(c, a, b)] = tidx
            index[(b, c, a)] = tidx

        # init regions per coordinate dictionary
        regions = {}
        # Sort each region in a coherent order, and substitude each triangle
        # by its index
        for i in range(4, len(self.coords)):
            v = useVertex[i][0][0]  # Get a vertex of a triangle
            r = []
            for _ in range(len(useVertex[i])):
                # Search the triangle beginning with vertex v
                t = [t for t in useVertex[i] if t[0] == v][0]
                r.append(index[t])  # Add the index of this triangle to region
                v = t[1]            # Choose the next vertex to search
            regions[i-4] = r        # Store region.

        return vor_coors, regions





class Delaunay2D_Torch:
    """
    Class to compute a Delaunay triangulation in 2D
    ref: http://en.wikipedia.org/wiki/Bowyer-Watson_algorithm
    ref: http://www.geom.uiuc.edu/~samuelp/del_project.html
    """

    def __init__(self, points, center=torch.tensor([0, 0]), radius=1000):
        """ Init and create a new frame to contain the triangulation
        center -- Optional position for the center of the frame. Default (0,0)
        radius -- Optional distance from corners to the center.
        """
        #center = np.asarray([0, 0])
        #center = torch.tensor(center)
        # Create coordinates for the corners of the frame
        #self.coords = [center+radius*np.array((-1, -1)),
        #               center+radius*np.array((+1, -1)),
        #               center+radius*np.array((+1, +1)),
        #               center+radius*np.array((-1, +1))]

        #self.coords = [torch.from_numpy(center+radius*np.array((-1, -1))),
        #               torch.from_numpy(center+radius*np.array((+1, -1))),
        #               torch.from_numpy(center+radius*np.array((+1, +1))),
        #               torch.from_numpy(center+radius*np.array((-1, +1)))]

        self.coords = [center+radius*torch.tensor((-1.0, -1.0)),
                       center+radius*torch.tensor((+1.0, -1.0)),
                       center+radius*torch.tensor((+1.0, +1.0)),
                       center+radius*torch.tensor((-1.0, +1.0))]

        # Create two dicts to store triangle neighbours and circumcircles.
        self.triangles = {}
        self.circles = {}

        # Create two CCW triangles for the frame
        T1 = (0, 1, 3)
        T2 = (2, 3, 1)
        self.triangles[T1] = [T2, None, None]
        self.triangles[T2] = [T1, None, None]

        # Compute circumcenters and circumradius for each triangle
        for t in self.triangles:
            self.circles[t] = self.circumcenter(t)

        for i,s in enumerate(points):
            #print(i)
            self.addPoint(s)

    def circumcenter(self, tri):
        """Compute circumcenter and circumradius of a triangle in 2D.
        Uses an extension of the method described here:
        http://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html
        """

        '''
        #pts_num = np.asarray([self.coords[v].numpy() for v in tri])
        pts_num = np.asarray([self.coords[v].numpy() for v in tri])
        pts2_num = np.dot(pts_num, pts_num.T)
        A_num = np.bmat([[2 * pts2_num, [[1.0],
                                 [1.0],
                                 [1.0]]],
                      [[[1.0, 1.0, 1.0, 0.0]]]])
        b_num = np.hstack((np.sum(pts_num * pts_num, axis=1), [1]))
        x_num = np.linalg.solve(A_num, b_num)
        bary_coords_num = x_num[:-1]
        center = np.dot(bary_coords_num, pts_num)
        radius = np.sum(np.square(pts_num[0] - center))

        '''


        pts = torch.vstack((self.coords[tri[0]], self.coords[tri[1]], self.coords[tri[2]]))
        pts2 = torch.mm(pts, pts.T)

        B = torch.tensor([[1.0], [1.0], [1.0] ])
        C = torch.hstack((2 * pts2,B))
        D = torch.tensor([1.0, 1.0, 1.0, 0.0])
        A = torch.vstack((C,D))


        
        b = torch.hstack((torch.sum(pts * pts, axis=1), torch.tensor([1])))
        b = b.reshape((4,1))
        #x = torch.mm(torch.pinverse(A), b)
        #try:

        x = torch.solve(b,A).solution

        #x_inv = torch.mm(torch.inverse(A),b)
        #except:
        #    print('Why')
        bary_coords = x[:-1]


        center1 = torch.dot(bary_coords[:,0], pts[:,0])
        center2 = torch.dot(bary_coords[:, 0], pts[:, 1])
        center = torch.hstack((center1,center2 ))

        # radius = np.linalg.norm(pts[0] - center) # euclidean distance

        radius = torch.sum(torch.square(pts[0] - center))# squared distance


        return (center, radius)

    def inCircleFast(self, tri, p):
        """Check if point p is inside of precomputed circumcircle of tri.
        """
        #center, radius = self.circles[tri]
        #return np.sum(np.square(center - p)) <= radius
        center, radius = self.circles[tri]
        return torch.sum(torch.square(center - p)) <= radius

    def inCircleRobust(self, tri, p):
        """Check if point p is inside of circumcircle around the triangle tri.
        This is a robust predicate, slower than compare distance to centers
        ref: http://www.cs.cmu.edu/~quake/robust.html
        """
        m1 = np.asarray([self.coords[v] - p for v in tri])
        m2 = np.sum(np.square(m1), axis=1).reshape((3, 1))
        m = np.hstack((m1, m2))    # The 3x3 matrix to check
        return np.linalg.det(m) <= 0

    def addPoint(self, p):
        """Add a point to the current DT, and refine it using Bowyer-Watson.
        """
        #p = p.numpy()
        #p = np.asarray(p)
        #p = p.numpy()
        idx = len(self.coords)
        # print("coords[", idx,"] ->",p)
        self.coords.append(p)

        # Search the triangle(s) whose circumcircle contains p
        bad_triangles = []
        for T in self.triangles:
            # Choose one method: inCircleRobust(T, p) or inCircleFast(T, p)
            if self.inCircleFast(T, p):
                bad_triangles.append(T)

        # Find the CCW boundary (star shape) of the bad triangles,
        # expressed as a list of edges (point pairs) and the opposite
        # triangle to each edge.
        boundary = []
        # Choose a "random" triangle and edge
        T = bad_triangles[0]
        edge = 0
        # get the opposite triangle of this edge
        while True:
            # Check if edge of triangle T is on the boundary...
            # if opposite triangle of this edge is external to the list
            tri_op = self.triangles[T][edge]
            if tri_op not in bad_triangles:
                # Insert edge and external triangle into boundary list
                boundary.append((T[(edge+1) % 3], T[(edge-1) % 3], tri_op))

                # Move to next CCW edge in this triangle
                edge = (edge + 1) % 3

                # Check if boundary is a closed loop
                if boundary[0][0] == boundary[-1][1]:
                    break
            else:
                # Move to next CCW edge in opposite triangle
                edge = (self.triangles[tri_op].index(T) + 1) % 3
                T = tri_op

        # Remove triangles too near of point p of our solution
        for T in bad_triangles:
            del self.triangles[T]
            del self.circles[T]

        # Retriangle the hole left by bad_triangles
        new_triangles = []
        for (e0, e1, tri_op) in boundary:
            # Create a new triangle using point p and edge extremes
            T = (idx, e0, e1)

            # Store circumcenter and circumradius of the triangle
            self.circles[T] = self.circumcenter(T)

            # Set opposite triangle of the edge as neighbour of T
            self.triangles[T] = [tri_op, None, None]

            # Try to set T as neighbour of the opposite triangle
            if tri_op:
                # search the neighbour of tri_op that use edge (e1, e0)

                for i, neigh in enumerate(self.triangles[tri_op]):
                    if neigh:
                        if e1 in neigh and e0 in neigh:
                                # change link to use our new triangle
                            self.triangles[tri_op][i] = T


            # Add triangle to a temporal list
            new_triangles.append(T)

        # Link the new triangles each another
        N = len(new_triangles)
        for i, T in enumerate(new_triangles):
            self.triangles[T][1] = new_triangles[(i+1) % N]   # next
            self.triangles[T][2] = new_triangles[(i-1) % N]   # previous

    def exportTriangles(self):
        """Export the current list of Delaunay triangles
        """
        # Filter out triangles with any vertex in the extended BBox
        return [(a-4, b-4, c-4)
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]

    def exportCircles(self):
        """Export the circumcircles as a list of (center, radius)
        """
        # Remember to compute circumcircles if not done before
        # for t in self.triangles:
        #     self.circles[t] = self.circumcenter(t)

        # Filter out triangles with any vertex in the extended BBox
        # Do sqrt of radius before of return
        return [(self.circles[(a, b, c)][0], sqrt(self.circles[(a, b, c)][1]))
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]

    def exportDT(self):
        """Export the current set of Delaunay coordinates and triangles.
        """
        # Filter out coordinates in the extended BBox
        coord = self.coords[4:]

        # Filter out triangles with any vertex in the extended BBox
        tris = [(a-4, b-4, c-4)
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]
        return coord, tris

    def exportExtendedDT(self):
        """Export the Extended Delaunay Triangulation (with the frame vertex).
        """
        return self.coords, list(self.triangles)


    def getVolume(self,f):
        cords, tris = self.exportDT()
        vertices = []

        tris = np.stack(tris)
        cords = torch.stack(cords)
        for i in range(len(tris)):
            vertices.append(cords[tris[i], :])

        sum_a = torch.tensor([0.0])
        for tri in tris:
            f_tri = f[tri]
            x_tri = cords[tri, :]
            sum_a += (0.5 * (((x_tri[1, 0] - x_tri[0, 0]) * (x_tri[2, 1] - x_tri[0, 1])) - (
                    (x_tri[1, 1] - x_tri[0, 1]) * (x_tri[2, 0] - x_tri[0, 0])))) * torch.mean(f_tri)

        return sum_a


    def plotDelaunay(self):
        import matplotlib.tri
        import matplotlib.collections
        cords, tris = self.exportDT()
        vertices = []

        tris = np.stack(tris)
        cords = torch.stack(cords)
        # Create a plot with matplotlib.pyplot
        fig, ax = plt.subplots()
        ax.margins(0.1)
        ax.set_aspect('equal')
        cords = cords.detach().clone().numpy()
        # Plot our Delaunay triangulation (plot in blue)
        cx, cy = zip(*cords)
        dt_tris = self.exportTriangles()
        ax.triplot(matplotlib.tri.Triangulation(cx, cy, dt_tris), 'bo--')
        plt.show()


'''

# Create a random set of 2D points
seeds = np.random.random((10, 2))

seeds_torch = torch.from_numpy(seeds)


# Create Delaunay Triangulation and insert points one by one
dt = Delaunay2D_Numpy()


dt_torch = Delaunay2D_Torch()

for s in seeds_torch:
    dt_torch.addPoint(s)


for s in seeds:
    dt.addPoint(s)

dt_tris = dt_torch.exportTriangles()
cords, tris = dt_torch.exportDT()
vertices = []

tris = np.stack(tris)
cords = torch.stack(cords)
for i in range(len(tris)):
    vertices.append(cords[tris[i],:])
import matplotlib.tri
import matplotlib.collections


# Create a plot with matplotlib.pyplot
fig, ax = plt.subplots()
ax.margins(0.1)
ax.set_aspect('equal')

# Plot our Delaunay triangulation (plot in blue)
cx, cy = zip(*seeds)
dt_tris = dt_torch.exportTriangles()
dt_tris2 = dt.exportTriangles()
ax.triplot(matplotlib.tri.Triangulation(cx, cy, dt_tris), 'bo--')
ax.triplot(matplotlib.tri.Triangulation(cx, cy, dt_tris2), 'bo--')
plt.show()

print(dt)
'''



def main():
    xyz = np.random.random((100, 3))
    area_underneath = trapezoidal_area(xyz)
    print(area_underneath)



def delaunayInt_Numpy(x, f):
    d = scipy.spatial.Delaunay(x)

    sum_a = 0
    for tri in d.vertices:
        f_tri = f[tri]
        x_tri = x[tri,:]
        sum_a += (0.5 * ( ((x_tri[1,0]-x_tri[0,0])*(x_tri[2,1]-x_tri[0,1])) - ((x_tri[1,1]-x_tri[0,1])*(x_tri[2,0]-x_tri[0,0]))  ))  *np.mean(f_tri)

    return sum_a



def delaunayInt_Torch_Mine(x,f):
    dt_torch = Delaunay2D_Torch(x)



    import time
    start_time = time.time()
    sum_a = dt_torch.getVolume(f)
    end_time = time.time()
    print(end_time-start_time)

    #cords, tris = dt_torch.exportDT()
    #vertices = []

    #tris = np.stack(tris)
    #cords = torch.stack(cords)
    #for i in range(len(tris)):
    #    vertices.append(cords[tris[i], :])

    #sum_a = torch.tensor([0.0])
    #for tri in tris:
    #    f_tri = f[tri]
    #    x_tri = x[tri, :]
    #    sum_a += (0.5 * (((x_tri[1, 0] - x_tri[0, 0]) * (x_tri[2, 1] - x_tri[0, 1])) - (
    #                (x_tri[1, 1] - x_tri[0, 1]) * (x_tri[2, 0] - x_tri[0, 0])))) * torch.mean(f_tri)


    return sum_a




class Delaunay_Constraint:
    def __init__(self, points, edges, holes=None, val = 'p'):
        self.points = points.cpu().numpy()
        self.edges = edges.cpu().numpy()

        t = {}
        t['vertices'] = self.points
        t['segments'] = self.edges
        if not holes==None:
            self.holes = holes.cpu().numpy()
            t['holes'] = self.holes
        #print('Starting Triangulating')
        if val == 'p':
            self.d = tr.triangulate(t, 'p')
        else:
            self.d = tr.triangulate(t)
        #print('Finished Triangulating')


    def getVolume(self,f):
        sum_a = torch.tensor([0.0])

        x_tri2 = self.points[self.d['triangles'], :]

        Je = ((((x_tri2[:,1, 0] - x_tri2[:,0, 0]) * (x_tri2[:,2, 1] - x_tri2[:,0, 1])) - (
                    (x_tri2[:,1, 1] - x_tri2[:,0, 1]) * (x_tri2[:,2, 0] - x_tri2[:,0, 0]))))
        Je = torch.from_numpy(Je).to(dev)

        f_tri2 = f[self.d['triangles']]
        f_tri2_mean = torch.mean(f_tri2,dim=1, dtype=torch.float64).to(dev)

        sum_a2 = torch.sum(0.5*(Je*f_tri2_mean[:,0]))
        #end_time = time.time()
        #print(end_time - start_time)
        return sum_a2.to(dev)


    def plotDelaunay(self):

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111)
        tr.plot(ax, **self.d)
        # ax.scatter(segments[:,0],segments[:,1])
        plt.show()



class Delaunay_Scipy:
    def __init__(self, points):
        self.points = points.cpu().numpy()
        self.d = scipy.spatial.Delaunay(self.points)


    def getVolume(self,f):
        sum_a = torch.tensor([0.0])

        #import time
        #start_time = time.time()
        #f = f.numpy()
        #for tri in self.d.vertices:
        #    f_tri = f[tri]
        #    x_tri = self.points[tri, :]
        #    sum_a += (0.5 * (((x_tri[1, 0] - x_tri[0, 0]) * (x_tri[2, 1] - x_tri[0, 1])) - (
        #            (x_tri[1, 1] - x_tri[0, 1]) * (x_tri[2, 0] - x_tri[0, 0])))) * torch.mean(f_tri,dtype=torch.float64)
        #end_time = time.time()
        #print(end_time - start_time)
        #start_time = time.time()
        x_tri2 = self.points[self.d.vertices, :]

        Je = ((((x_tri2[:,1, 0] - x_tri2[:,0, 0]) * (x_tri2[:,2, 1] - x_tri2[:,0, 1])) - (
                    (x_tri2[:,1, 1] - x_tri2[:,0, 1]) * (x_tri2[:,2, 0] - x_tri2[:,0, 0]))))

        Je = torch.from_numpy(Je).to(dev)
        f_tri2 = f[self.d.vertices]
        f_tri2_mean = torch.mean(f_tri2,dim=1, dtype=torch.float64).to(dev)

        sum_a2 = torch.sum(0.5*(Je*f_tri2_mean[:,0]))
        #end_time = time.time()
        #print(end_time - start_time)
        return sum_a2.to(dev)


    def plotDelaunay(self):


        plt.triplot(self.points[:, 0], self.points[:, 1], self.d.simplices)

        plt.plot(self.points[:, 0], self.points[:, 1], 'o')

        plt.show()





def delaunayInt_Torch(x,f):

    x = x.numpy()
    f = f.numpy()
    d = scipy.spatial.Delaunay(x[:, :2])

    sum_a = 0
    for tri in d.vertices:
        f_tri = f[tri]
        x_tri = x[tri, :]
        sum_a += (0.5 * (((x_tri[1, 0] - x_tri[0, 0]) * (x_tri[2, 1] - x_tri[0, 1])) - (
                    (x_tri[1, 1] - x_tri[0, 1]) * (x_tri[2, 0] - x_tri[0, 0])))) * np.mean(f_tri)


    return torch.tensor(sum_a)


def trapezoidal_area(xyz):
    """Calculate volume under a surface defined by irregularly spaced points
    using delaunay triangulation. "x,y,z" is a <numpoints x 3> shaped ndarray."""
    d = scipy.spatial.Delaunay(xyz[:,:2])
    tri = xyz[d.vertices]

    a = tri[:,0,:2] - tri[:,1,:2]
    b = tri[:,0,:2] - tri[:,2,:2]
    proj_area = np.cross(a, b).sum(axis=-1)
    zavg = tri[:,:,2].sum(axis=1)
    vol = zavg * np.abs(proj_area) / 6.0
    return vol.sum()


def trapezoidal_area(xyz):
    """Calculate volume under a surface defined by irregularly spaced points
    using delaunay triangulation. "x,y,z" is a <numpoints x 3> shaped ndarray."""
    d = scipy.spatial.Delaunay(xyz[:,:2])
    tri = xyz[d.vertices]

    a = tri[:,0,:2] - tri[:,1,:2]
    b = tri[:,0,:2] - tri[:,2,:2]
    proj_area = np.cross(a, b).sum(axis=-1)
    zavg = tri[:,:,2].sum(axis=1)
    vol = zavg * np.abs(proj_area) / 6.0
    return vol.sum()



def voronoi_volumes(points):
    v = scipy.spatial.Voronoi(points)



    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = scipy.spatial.ConvexHull(v.vertices[indices]).volume
    return vol

def voronoi_finitevolumes(points):
    v = scipy.spatial.Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(v,radius=0.0)

    #pts = MultiPoint([Point(i) for i in points])
    #mask = pts.convex_hull
    # = []
    #for region in regions:
    #    polygon = vertices[region]
    #    shape = list(polygon.shape)
    #    shape[0] += 1
    #    p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
    #    poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
    #    new_vertices.append(poly)
    #    plt.fill(*zip(*poly), "brown", alpha=0.4, edgecolor='black')


    #plt.show()


    vol = np.zeros(v.npoints)

    for i, reg_num in enumerate(regions):
        indices = reg_num
        if -1 in indices:  # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = scipy.spatial.ConvexHull(vertices[indices]).volume

            plt.scatter(vertices[indices][:, 0], vertices[indices][:, 1])
            plt.show()
    return np.sum(vol)




def voronoi_twoDim(x,f):
    """Calculate volume under a surface defined by irregularly spaced points
    using delaunay triangulation. "x,y,z" is a <numpoints x 3> shaped ndarray."""
    a = voronoi_volumes(x)
    notInf = np.where(~np.isinf(a) )

    a_sum = np.sum(a[notInf])






def square_voronoi(xy, bbox): #bbox: (min_x, max_x, min_y, max_y)
    # Select points inside the bounding box
    points_center = xy[np.where((bbox[0] <= xy[:,0]) * (xy[:,0] <= bbox[1]) * (bbox[2] <= xy[:,1]) * (bbox[2] <= bbox[3]))]
    # Mirror points
    points_left = np.copy(points_center)
    points_left[:, 0] = bbox[0] - (points_left[:, 0] - bbox[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bbox[1] + (bbox[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bbox[2] - (points_down[:, 1] - bbox[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bbox[3] + (bbox[3] - points_up[:, 1])
    points = np.concatenate((points_center, points_left, points_right, points_down, points_up,), axis=0)
    # Compute Voronoi
    vor = scipy.spatial.Voronoi(points)
    # Filter regions (center points should* be guaranteed to have a valid region)
    # center points should come first and not change in size
    regions = [vor.regions[vor.point_region[i]] for i in range(len(points_center))]
    vor.filtered_points = points_center
    vor.filtered_regions = regions
    return vor

#also stolen from: https://stackoverflow.com/questions/28665491/getting-a-bounded-polygon-coordinates-from-voronoi-cells
def area_region(vertices):
    # Polygon's signed area
    A = 0
    for i in range(0, len(vertices) - 1):
        s = (vertices[i, 0] * vertices[i + 1, 1] - vertices[i + 1, 0] * vertices[i, 1])
        A = A + s
    return np.abs(0.5 * A)

def f(x,y):
    return np.cos(10*x*y) * np.exp(-x**2 - y**2)


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)





