import numpy as np

def weight_func(X, Y, Z, p, epsilon):
    tmp = (X - p[0]) **2 + (Y - p[1]) **2 + (Z - p[2]) **2 + epsilon **2
    return 1 / tmp

def constraint(X, Y, Z, p, p_normal, phi0 = 0):
    tmp = (X - p[0]) * p_normal[0] + (Y - p[1]) * p_normal[1] + (Z - p[2]) * p_normal[2] + phi0
    return tmp

def main():
    # LOAD DATA
    point_clouds = np.load("../data/point_clouds.npy")
    point_cloud_normals = np.load( "../data/point_cloud_normals.npy")
    N = point_clouds.shape[0]

    # 本方法不需要扩增points
    # CONSTRUCT GRID
    X, Y, Z = np.mgrid[-0.65:0.65:100j, -0.65:0.65:100j, -0.65:0.65:100j]
    
    sdf = np.zeros_like(X)
    w_sum = np.zeros_like(X)
    epsilon = 0.000001

    # 遍历点云
    for i in range(N):
        print(i)
        wi = weight_func(X, Y, Z, point_clouds[i], epsilon)
        Si = constraint(X, Y, Z, point_clouds[i], point_cloud_normals[i], phi0=0)
        w_sum += wi **2 
        sdf += Si * wi **2

    sdf *= 1 / w_sum

    print(sdf.shape)
    print(sdf)
    np.save('sdf.npy', sdf)


        

if __name__ == "__main__":
    main()