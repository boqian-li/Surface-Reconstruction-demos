import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg

np.random.seed(0)

def print_iteration_info(x):
    print(f"Iteration: {print_iteration_info.iteration}")
    print_iteration_info.iteration += 1

def conjugate_grad(A, b, epsilon=1e-10, maxiter = 1000):
    n = A.shape[0]
    x = np.zeros(n)
    r = b - A @ x
    q = r.copy()
    r_old = np.inner(r, r)
    for it in range(maxiter):
        alpha = r_old / np.inner(q, A @ q)
        x += alpha * q
        r -= alpha * A @ q
        r_new = np.inner(r, r)
        print(np.sqrt(r_new))
        if np.sqrt(r_new) < epsilon:
            break
        beta = r_new / r_old
        q = r + beta * q
        r_old = r_new.copy()
    return x

def get_points(points, normals, d=0.01, sample = None):
    '''
    Get internal, external and ori points
    '''
    if sample != None:
        sample_num = int(sample * points.shape[0])
        full_num = points.shape[0]
        sample_indices = np.random.choice(full_num, sample_num, replace=False)
        
        points = points[sample_indices, :]
        normals = normals[sample_indices, :]

    more_points = points.copy()
    more_points = np.concatenate((more_points, points + normals * d), axis=0)
    more_points = np.concatenate((more_points, points - normals * d), axis=0)
    assert more_points.shape[0] == 3*points.shape[0]

    return more_points


def distance(X1, Y1, Z1, X2, Y2, Z2):
    return np.sqrt((X1 - X2)**2 + (Y1 - Y2)**2 + (Z1 - Z2)**2)

class Basic_function:
    def __init__(self, name='biharmonic_spline', c=None):
        assert name in ['biharmonic_spline', 'thin_plate_spline', 'gaussian', 'multiquadric']
        self.name = name
        self.c = c

    def __call__(self, X1, Y1, Z1, X2, Y2, Z2):
        r = distance(X1, Y1, Z1, X2, Y2, Z2)
        if self.name == 'biharmonic_spline':
            return r
        elif self.name == 'thin_plate_spline':  
            return r**2 * np.log10(r)
        elif self.name == 'gaussian':
            return np.exp(-self.c * r**2)
        elif self.name == 'multiquadric':
            return np.sqrt(r**2 + self.c**2)

class Interpolator:
    def __init__(self, x, points, basic_function):
        self.x = x
        self.points = points
        self.func = basic_function

    def __call__(self, X, Y, Z):
        sdf = np.ones_like(X)

        sdf *= self.x[-4]
        sdf += X * self.x[-3] + Y * self.x[-2] + Z * self.x[-1]
        for i, point in enumerate(self.points):
            X1 = np.full(X.shape, point[0])
            Y1 = np.full(X.shape, point[1])
            Z1 = np.full(X.shape, point[2])
            sdf += self.x[i] * self.func(X, Y, Z, X1, Y1, Z1)
            print(i)
        return sdf



def main():
    # LOAD DATA
    point_clouds = np.load("point_clouds.npy")
    point_cloud_normals = np.load( "point_cloud_normals.npy")

    # GET FULL POINTS
    d = 0.01
    #sample = 1.0
    points = get_points(point_clouds, point_cloud_normals, d=d)

    # visualise
    N = points.shape[0]
    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(point_clouds[:, 0], point_clouds[:, 1], point_clouds[:, 2], c="k", s=0.1)
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(points[:int(N/3), 0], points[:int(N/3), 1], points[:int(N/3), 2], c="k", s=0.1)
    plt.show()
    plt.close("all")    

    # construct A, b
    A = np.zeros((N, N))
    b = np.zeros(N+4)

    # for b
    for i in range(N):
        if i < int(N / 3):
            b[i] = 0
        elif i < int(N * 2 / 3):
            b[i] = d
        else:
            b[i] = -d
    
    # for A
    print(1111111111)
    basic_function = Basic_function('gaussian', c = 6000)
    X1 = np.tile(points[:, 0], (N, 1))
    Y1 = np.tile(points[:, 1], (N, 1))
    Z1 = np.tile(points[:, 2], (N, 1))
    X2 = np.transpose(X1)
    Y2 = np.transpose(Y1)
    Z2 = np.transpose(Z1)
    A += basic_function(X1, Y1, Z1, X2, Y2, Z2)
    print(22222222222222)

    tmp = np.insert(points, 0, np.ones(N), axis = 1)
    A = np.concatenate((np.concatenate((A, tmp), axis=1), np.concatenate((np.transpose(tmp), np.zeros((4, 4))), axis=1)), axis=0)

    # solve out 
    #x = conjugate_grad(A, b, epsilon = 1e-5, maxiter=2000)

    print_iteration_info.iteration = 0
    sparse_A = csr_matrix(A)
    x, info = cg(sparse_A, b, tol = 1e-5, maxiter=3000, callback=print_iteration_info)
    # tol参数不好控制，可以通过调整maxiter来搞，使得最后的残差的模在指定范围内，而这个指定范围需要自己探索，太少了可能噪声多，太大了细节不丰富。
    print("Total Iterations:", print_iteration_info.iteration)
    print("residual final:", np.linalg.norm(b - sparse_A.dot(x)))
    with open("output.txt", "w") as file:
        file.write("residual final:" + str(np.linalg.norm(b - sparse_A.dot(x))) + "\n")
        file.write("Total Iterations: " + str(print_iteration_info.iteration) + "\n")


    interpolator = Interpolator(x, points, basic_function)

    # gird 
    X, Y, Z = np.mgrid[-0.65:0.65:100j, -0.65:0.65:100j, -0.65:0.65:100j]
    print(3333333333333333333)
    sdf = interpolator(X, Y, Z)

    print(sdf.shape)
    print(sdf)
    np.save('sdf.npy', sdf)
   
        

if __name__ == "__main__":
    main()




