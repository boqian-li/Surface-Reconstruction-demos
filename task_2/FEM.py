import tetgen
import pyvista
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from scipy.spatial import cKDTree
import open3d as o3d
import mcubes

def remove_duplicate_rows(array):
    _, indices, counts = np.unique(array, axis=0, return_index=True, return_counts=True)
    unique_indices = indices[counts == 1]
    return unique_indices

def extract_surface(tets):
    triangles = []
    for tet in tets:
        triangles.append(np.array([tet[0], tet[2], tet[1]]))
        triangles.append(np.array([tet[0], tet[1], tet[3]]))
        triangles.append(np.array([tet[1], tet[2], tet[3]]))
        triangles.append(np.array([tet[0], tet[3], tet[2]]))
    
    triangles = np.array(triangles)
    new_triangles = triangles[remove_duplicate_rows(np.sort(triangles, axis=1))]

    return new_triangles
  

def remove_unused_rows(index_array, data_array):
    '''
    清除无用行，更新索引数组
    '''
    # 获取数据数组的行数
    num_rows = data_array.shape[0]
    new_index_array = np.zeros_like(index_array)

    # 创建布尔掩码，标记被索引到的行
    mask = np.zeros(num_rows, dtype=bool)
    for indices in index_array:
        for i in indices:
            mask[i] = True

    # 根据掩码筛选出被索引到的行
    new_data_array = data_array[mask]

    # 更新索引数组，根据筛选后的行重新生成新的索引
    M = new_index_array.shape[0]
    tmp = np.cumsum(mask) - 1
    for i in range(M):
        new_index_array[i] = tmp[list(index_array[i])]

    return new_index_array, new_data_array


##############################  预处理部分    ############################
print("======== START PRE-PROCESS ======")
# LOAD DATA
point_clouds = np.load("../data/point_clouds.npy")
point_cloud_normals = np.load("../data/point_cloud_normals.npy")
corr = np.load("../data/corr.npy")
target_mesh = trimesh.load("../data/target.obj")
target_all = target_mesh.vertices.copy()

reader = pyvista.get_reader("../target/tmp_1.obj")
mesh_ = reader.read()
tgen = tetgen.TetGen(mesh_)
vertices, tets = tgen.tetrahedralize() #有很多参数可以调
# print(vertices.shape) #(113801, 3) 一共113801个顶点
# print(tets.shape) #(442747, 4) 一共有442747个四面体，每个存储了四个顶点在nodes数组中的index

# GET KEY POINTS
key_indices = []
key_tarpoints = []
tree = cKDTree(vertices)
key_fullroad = []

for i, j in corr:
    distances, closest_point_index = tree.query(point_clouds[i])
    key_indices.append(closest_point_index)
    key_tarpoints.append(target_all[j])
    key_fullroad.append(target_all[j] - vertices[closest_point_index])

key_indices = np.array(key_indices) 
key_tarpoints = np.array(key_tarpoints)
key_fullroad = np.array(key_fullroad) # norm ~ 0.25 for each
print("======== END PRE-PROCESS ========")

#################################  算法部分  ###################################
###  定义各参数 and 所需变量  ###
dt = 0.0005
damp = 0.99
mass = 1
k_ = 50000  # Young's modulus
v_ = 0.1  # Poisson's ratio
mu_ = k_ / (2 * (1 + v_))
lambda_ = k_ * v_ / ((1 + v_) * (1 - 2*v_))
STEP_NUM = 100
DAMP_STEP_NUM = 50
key_step = key_fullroad / STEP_NUM  #每一次更新，key points的移动向量

V = np.zeros_like(vertices)
#vertices = X
Force = np.zeros_like(vertices)
Bm_all = []
W_all = np.zeros(tets.shape[0])

###  初始化变量  ###
for tet in range(tets.shape[0]):
    Dm = []
    Dm.append(np.array([vertices[tets[tet, 0], 0] - vertices[tets[tet, 3], 0], 
                        vertices[tets[tet, 1], 0] - vertices[tets[tet, 3], 0],
                        vertices[tets[tet, 2], 0] - vertices[tets[tet, 3], 0]]))
    Dm.append(np.array([vertices[tets[tet, 0], 1] - vertices[tets[tet, 3], 1], 
                        vertices[tets[tet, 1], 1] - vertices[tets[tet, 3], 1],
                        vertices[tets[tet, 2], 1] - vertices[tets[tet, 3], 1]]))
    Dm.append(np.array([vertices[tets[tet, 0], 2] - vertices[tets[tet, 3], 2], 
                        vertices[tets[tet, 1], 2] - vertices[tets[tet, 3], 2],
                        vertices[tets[tet, 2], 2] - vertices[tets[tet, 3], 2]]))
    Dm = np.array(Dm)
    Bm_all.append(np.linalg.inv(Dm))
    W_all[tet] = np.linalg.det(Dm) / 6

Bm_all = np.array(Bm_all)

###  循环Update ---- key point moving process  ###
for STEP in range(STEP_NUM):
    print(STEP)
    Force[...] = 0.0

    ##  可视化  ##
    if STEP % 5 == 0:
        triangles = extract_surface(tets)
        new_triangles, surface_vertices = remove_unused_rows(triangles, vertices)
        mcubes.export_obj(surface_vertices, new_triangles, '123.obj')
        mesh= o3d.io.read_triangle_mesh('123.obj')
        o3d.visualization.draw_geometries([mesh])
    

    ## 遍历四面体，计算各个顶点的力 ##
    for tet in range(tets.shape[0]):
        Ds = []
        Ds.append(np.array([vertices[tets[tet, 0], 0] - vertices[tets[tet, 3], 0], 
                            vertices[tets[tet, 1], 0] - vertices[tets[tet, 3], 0],
                            vertices[tets[tet, 2], 0] - vertices[tets[tet, 3], 0]]))
        Ds.append(np.array([vertices[tets[tet, 0], 1] - vertices[tets[tet, 3], 1], 
                            vertices[tets[tet, 1], 1] - vertices[tets[tet, 3], 1],
                            vertices[tets[tet, 2], 1] - vertices[tets[tet, 3], 1]]))
        Ds.append(np.array([vertices[tets[tet, 0], 2] - vertices[tets[tet, 3], 2], 
                            vertices[tets[tet, 1], 2] - vertices[tets[tet, 3], 2],
                            vertices[tets[tet, 2], 2] - vertices[tets[tet, 3], 2]]))
        Ds = np.array(Ds)
        F = Ds @ Bm_all[tet]
        I = np.eye(F.shape[0])
        P = mu_*(F + np.transpose(F) - 2*I) + lambda_ * np.trace(F - I) * I
        H = - W_all[tet] * P @ np.transpose(Bm_all[tet])
        Force[tets[tet, 0]] += H[:, 0]
        Force[tets[tet, 1]] += H[:, 1]
        Force[tets[tet, 2]] += H[:, 2]
        Force[tets[tet, 3]] += -H[:, 0] - H[:, 1] - H[:, 2]

    ## 遍历更新顶点速度、位置（key points无需更新） ##
    for p in range(vertices.shape[0]):
        if p in key_indices: continue
        V[p] += dt * Force[p] / mass
        V[p] *= damp
        vertices[p] += V[p] * dt

    ## 强制拉取，更新key points 位置 ##
    for i, p in enumerate(key_indices):
        vertices[p] += key_step[i]

    np.savetxt('vertices.txt', vertices, fmt='%f', delimiter=',')
    np.savetxt('V.txt', V, fmt='%f', delimiter=',')
    np.savetxt('Force.txt', Force, fmt='%f', delimiter=',')

    ### 循环Update ---- damping process ###
    for DAMP_STEP in range(10):
        Force[...] = 0.0
        ## 遍历四面体，计算各个顶点的力 ##
        for tet in range(tets.shape[0]):
            Ds = []
            Ds.append(np.array([vertices[tets[tet, 0], 0] - vertices[tets[tet, 3], 0], 
                                vertices[tets[tet, 1], 0] - vertices[tets[tet, 3], 0],
                                vertices[tets[tet, 2], 0] - vertices[tets[tet, 3], 0]]))
            Ds.append(np.array([vertices[tets[tet, 0], 1] - vertices[tets[tet, 3], 1], 
                                vertices[tets[tet, 1], 1] - vertices[tets[tet, 3], 1],
                                vertices[tets[tet, 2], 1] - vertices[tets[tet, 3], 1]]))
            Ds.append(np.array([vertices[tets[tet, 0], 2] - vertices[tets[tet, 3], 2], 
                                vertices[tets[tet, 1], 2] - vertices[tets[tet, 3], 2],
                                vertices[tets[tet, 2], 2] - vertices[tets[tet, 3], 2]]))
            Ds = np.array(Ds)
            F = Ds @ Bm_all[tet]
            I = np.eye(F.shape[0])
            P = mu_*(F + np.transpose(F) - 2*I) + lambda_ * np.trace(F - I) * I
            H = - W_all[tet] * P @ np.transpose(Bm_all[tet])
            Force[tets[tet, 0]] += H[:, 0]
            Force[tets[tet, 1]] += H[:, 1]
            Force[tets[tet, 2]] += H[:, 2]
            Force[tets[tet, 3]] += -H[:, 0] - H[:, 1] - H[:, 2]

        ## 遍历更新顶点速度、位置（key points无需更新） ##
        for p in range(vertices.shape[0]):
            if p in key_indices: continue
            V[p] += dt * Force[p] / mass
            V[p] *= damp
            vertices[p] += V[p] * dt


### 循环Update ---- damping process ###
damp = 0.7
for STEP in range(DAMP_STEP_NUM):
    Force[...] = 0.0
    ## 遍历四面体，计算各个顶点的力 ##
    for tet in range(tets.shape[0]):
        Ds = []
        Ds.append(np.array([vertices[tets[tet, 0], 0] - vertices[tets[tet, 3], 0], 
                            vertices[tets[tet, 1], 0] - vertices[tets[tet, 3], 0],
                            vertices[tets[tet, 2], 0] - vertices[tets[tet, 3], 0]]))
        Ds.append(np.array([vertices[tets[tet, 0], 1] - vertices[tets[tet, 3], 1], 
                            vertices[tets[tet, 1], 1] - vertices[tets[tet, 3], 1],
                            vertices[tets[tet, 2], 1] - vertices[tets[tet, 3], 1]]))
        Ds.append(np.array([vertices[tets[tet, 0], 2] - vertices[tets[tet, 3], 2], 
                            vertices[tets[tet, 1], 2] - vertices[tets[tet, 3], 2],
                            vertices[tets[tet, 2], 2] - vertices[tets[tet, 3], 2]]))
        Ds = np.array(Ds)
        F = Ds @ Bm_all[tet]
        I = np.eye(F.shape[0])
        P = mu_*(F + np.transpose(F) - 2*I) + lambda_ * np.trace(F - I) * I
        H = - W_all[tet] * P @ np.transpose(Bm_all[tet])
        Force[tets[tet, 0]] += H[:, 0]
        Force[tets[tet, 1]] += H[:, 1]
        Force[tets[tet, 2]] += H[:, 2]
        Force[tets[tet, 3]] += -H[:, 0] - H[:, 1] - H[:, 2]

    ## 遍历更新顶点速度、位置（key points无需更新） ##
    for p in range(vertices.shape[0]):
        if p in key_indices: continue
        V[p] += dt * Force[p] / mass
        V[p] *= damp
        vertices[p] += V[p] * dt

###############################  后处理部分  ################################
