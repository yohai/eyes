# %%
import plotly.graph_objects as go
import numpy as np
from stl import mesh
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def transform_mesh_using_pca(stl_file_path):
    original_mesh = mesh.Mesh.from_file(stl_file_path)
    vertices = np.vstack([original_mesh.v0,
                          original_mesh.v1,
                          original_mesh.v2])
    pca = PCA(n_components=3)
    pca.fit(vertices)
    transformed_vertices = vertices.dot(pca.components_.T)

    return vertices


def plot_point_cloud(vs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = vs.T
    # Plot point cloud
    ax.plot(x, y, z, '.')

    # Plot PCA planes
    lx = (x.max() - x.min())/1.5
    ly = (y.max() - y.min())/1.5
    lz = (z.max() - z.min())/1.5
    xx, yy = np.meshgrid(np.linspace(-lx, lx, 5), np.linspace(-ly, ly, 5))
    ax.plot_surface(xx, yy, 0 * xx, alpha=0.25, color='r')
    yy, zz = np.meshgrid(np.linspace(-ly, ly, 5), np.linspace(-lz, lz, 5))
    ax.plot_surface(0*xx, yy, zz, alpha=0.25, color='g')
    zz, xx = np.meshgrid(np.linspace(-lz, lz, 5), np.linspace(-lx, lx, 5))
    ax.plot_surface(xx, 0 * xx, zz, alpha=0.25, color='m')
    ax.set_aspect('equal')
    # ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
    plt.show()


original_mesh = mesh.Mesh.from_file('scans/example1.stl')


def stl2mesh3d(stl_mesh):
    # stl_mesh is read by nympy-stl from a stl file; it is  an array of faces/triangles (i.e. three 3d points)
    # this function extracts the unique vertices and the lists I, J, K to define a Plotly mesh3d
    p, q, r = stl_mesh.vectors.shape  # (p, 3, 3)
    # the array stl_mesh.vectors.reshape(p*q, r) can contain multiple copies of the same vertex;
    # extract unique vertices from all mesh triangles
    vertices, ixr = np.unique(stl_mesh.vectors.reshape(
        p*q, r), return_inverse=True, axis=0)
    x, y, z = vertices.T
    I = np.take(ixr, [3*k for k in range(p)])
    J = np.take(ixr, [3*k+1 for k in range(p)])
    K = np.take(ixr, [3*k+2 for k in range(p)])
    return x, y, z, I, J, K


def plot_mesh_with_plotly(stl_mesh):
    x, y, z, i, j, k = stl2mesh3d(stl_mesh)

    fig = go.Figure(data=[
        go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k)
    ])

    fig.show()

# # Usage example
# plot_mesh_with_plotly(original_mesh)


# # Usage
# vertices = transform_mesh_using_pca('scans/example1.stl')
# # plot_point_cloud(vertices)
# #%%
# vertices, I, J, K = stl2mesh3d(original_mesh)
