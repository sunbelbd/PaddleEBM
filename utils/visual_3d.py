#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#   
#   Author: Yifei Xu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import paddle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from functools import reduce
from skimage import measure
from mpl_toolkits.mplot3d import axes3d, Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

def visualize_voxel(voxels, output_filename=None):

    fig = plt.figure(figsize=(16, 8))
    voxels = voxels.numpy()[:8]
    for i, voxel in enumerate(voxels):

        ax = plt.subplot(2, 4, i+1, projection='3d')
        plt.subplots_adjust(0,0,1,1,0,0)
        # ax.voxels(voxel)
        verts, faces, normals, values = measure.marching_cubes(voxel > 0.5, 0)
        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor((0,0,0,0.3))
        ax.add_collection3d(mesh)
        side = 4
        ax.set_xlim(0 + side, voxel.shape[0] - side)
        ax.set_ylim(0 + side, voxel.shape[0] - side)
        ax.set_zlim(0 + side, voxel.shape[0] - side)
        ax.axis("off")
        fig.tight_layout()
    if output_filename is not None:
        plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)

def visualize_pcd(point_clouds, num_rows=3, num_cols=3, output_filename=None, mode=1, idx=None):
    """"Converts a mini-batch point cloud into a numpy image array.

    Parameters:
        point_clouds (tensor) --  the input point cloud array
        image_num (int)      --  the convert iamge numbers
        imtype (type)        --  the desired type of the converted numpy array
    """

    if point_clouds.shape[1] < 10: 
        point_clouds = np.swapaxes(point_clouds, 1, 2)
    num_clouds = len(point_clouds)
    # num_rows = min(num_rows, num_clouds // num_cols + 1)
    if mode == 0: 
        fig = plt.figure(figsize=(num_cols * 4, num_rows * 4))
        for i, pts in enumerate(point_clouds[:num_cols*num_rows]):
            if point_clouds.shape[2] == 3: 
                ax = plt.subplot(num_rows, num_cols, i+1, projection='3d')
                plt.subplots_adjust(0,0,1,1,0,0)
                #ax.axis('off')
                if idx is not None:
                    ax.set_title(str(idx[i]))
                ax.scatter(pts[:,0], pts[:,2], pts[:,1], marker='.', s=50, c=pts[:,2], cmap=plt.get_cmap('gist_rainbow'))
            else: 
                ax = plt.subplot(num_rows, num_cols, i+1)
                plt.subplots_adjust(0,0,1,1,0,0)
                # ax.axis('off')
                if idx is not None:
                    ax.set_title(str(idx[i]))
                ax.scatter(pts[:,1], -pts[:,0], marker='.', s=30)
        if output_filename is not None:
            plt.savefig(output_filename, bbox_inches='tight')
        return fig
    elif mode == 1: 
        point_clouds = point_clouds[..., [0,2,1]]
        row_imgs = []
        for ir in range(num_rows):
            col_imgs = []
            for ic in range(num_cols):
                idx = ir * num_cols + ic
                col_imgs.append(draw_point_cloud(point_clouds[idx], zrot=80 / 180.0 * np.pi,
                                    xrot=-45 / 180.0 * np.pi, yrot=-20 / 180.0 * np.pi)
                                if idx < point_clouds.shape[0] else np.zeros((500, 500)))
            row_imgs.append(np.concatenate(col_imgs, axis=1))
        im_array = np.concatenate(row_imgs, axis=0)
        img = Image.fromarray(np.uint8(im_array * 255.0))
        if output_filename is not None:
            img.save(output_filename)
        return np.array(img)
    elif mode == 3: 
        assert output_filename is not None, "in mode 3, output filename should not be None."
        if not os.path.exists(output_filename):
            os.makedirs(output_filename)
        for i, pts in enumerate(point_clouds):
            img = draw_point_cloud(point_clouds[i],
                                zrot=80 / 180.0 * np.pi,
                                xrot=-45 / 180.0 * np.pi,
                                yrot=-20 / 180.0 * np.pi)
            img = Image.fromarray(np.uint8(img * 255.0))
            img.save('%s/%d.png' % (output_filename, i))

def draw_point_cloud(input_points, canvasSize=500, space=200, diameter=25,
                     xrot=0, yrot=0, zrot=0, switch_xyz=[0, 1, 2], normalize=True):
    """ Render point cloud to image with alpha channel.
        Input:
            points: Nx3 numpy array (+y is up direction)
        Output:
            gray image as numpy array of size canvasSizexcanvasSize
    """
    image = np.zeros((canvasSize, canvasSize))
    if input_points is None or input_points.shape[0] == 0:
        return image

    points = input_points[:, switch_xyz]
    M = euler2mat(zrot, yrot, xrot)
    points = (np.dot(M, points.transpose())).transpose()

    # Normalize the point cloud
    # We normalize scale to fit points in a unit sphere
    if normalize:
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points) ** 2, axis=-1)))
        points /= furthest_distance

    # Pre-compute the Gaussian disk
    radius = (diameter - 1) / 2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) * (i - radius) + (j - radius) * (j - radius) <= radius * radius:
                disk[i, j] = np.exp((-(i - radius) ** 2 - (j - radius) ** 2) / (radius ** 2))
    mask = np.argwhere(disk > 0)
    dx = mask[:, 0]
    dy = mask[:, 1]
    dv = disk[disk > 0]

    # Order points by z-buffer
    zorder = np.argsort(points[:, 2])
    points = points[zorder, :]
    points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2] - np.min(points[:, 2])))
    max_depth = np.max(points[:, 2])

    for i in range(points.shape[0]):
        j = points.shape[0] - i - 1
        x = points[j, 0]
        y = points[j, 1]
        xc = canvasSize / 2 + (x * space)
        yc = canvasSize / 2 + (y * space)
        xc = int(np.round(xc))
        yc = int(np.round(yc))

        px = dx + xc
        py = dy + yc

        px[px < 0] = 0
        px[px > image.shape[0]-1] = 0
        py[py < 0] = 0
        py[py > image.shape[1]-1] = 0
        image[px, py] = image[px, py] * 0.7 + dv * (max_depth - points[j, 2]) * 0.3
    image[0,0] = 0
    image = image / np.max(image)
    return image


def euler2mat(z=0, y=0, x=0):
    ''' Return matrix for rotations around z, y and x axes

    Uses the z, then y, then x convention above

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Returns
    -------
    M : array shape (3,3)
       Rotation matrix giving same rotation as for given angles

    Examples
    --------
    >>> zrot = 1.3 # radians
    >>> yrot = -0.1
    >>> xrot = 0.2
    >>> M = euler2mat(zrot, yrot, xrot)
    >>> M.shape == (3, 3)
    True

    The output rotation matrix is equal to the composition of the
    individual rotations

    >>> M1 = euler2mat(zrot)
    >>> M2 = euler2mat(0, yrot)
    >>> M3 = euler2mat(0, 0, xrot)
    >>> composed_M = np.dot(M3, np.dot(M2, M1))
    >>> np.allclose(M, composed_M)
    True

    You can specify rotations by named arguments

    >>> np.all(M3 == euler2mat(x=xrot))
    True

    When applying M to a vector, the vector should column vector to the
    right of M.  If the right hand side is a 2D array rather than a
    vector, then each column of the 2D array represents a vector.

    >>> vec = np.array([1, 0, 0]).reshape((3,1))
    >>> v2 = np.dot(M, vec)
    >>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
    >>> vecs2 = np.dot(M, vecs)

    Rotations are counter-clockwise.

    >>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
    >>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
    True
    >>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
    >>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
    True
    >>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
    >>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
    True

    Notes
    -----
    The direction of rotation is given by the right-hand rule (orient
    the thumb of the right hand along the axis around which the rotation
    occurs, with the end of the thumb at the positive end of the axis;
    curl your fingers; the direction your fingers curl is the direction
    of rotation).  Therefore, the rotations are counterclockwise if
    looking along the axis of rotation from positive to negative.
    '''
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
                [[cosy, 0, siny],
                 [0, 1, 0],
                 [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
                [[1, 0, 0],
                 [0, cosx, -sinx],
                 [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)
