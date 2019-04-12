import numpy as np
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import axes3d

n_samples = 300

x = np.random.rand(n_samples)
y = np.random.rand(n_samples)
z = np.random.rand(n_samples)

ax = mp.gca(projection = "3d")

mp.title('Scatter 3D', fontsize=20)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.set_zlabel('z', fontsize=14)
mp.tick_params(labelsize=10)

mp.gca().scatter(x,y,z,c = np.array([x,y,z]).T,
                 s=100 * np.linalg.norm((x, y, z), axis=0))
mp.show()


