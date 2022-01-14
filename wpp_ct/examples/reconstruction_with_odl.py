


import odl

import numpy as np
import matplotlib.pyplot as plt 


reco_space = odl.uniform_discr(
            min_pt=[-256, -256], max_pt=[256, 256], shape=[512, 512])

phantom = odl.phantom.shepp_logan(reco_space, modified=True)

geometry = odl.tomo.parallel_beam_geometry(reco_space, 30)
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

sinogram = ray_trafo(phantom)

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.imshow(phantom)
ax1.set_title("Shepp Logan Phantom")

ax2.imshow(sinogram)
ax2.set_title("Sinogram")
ax2.set_xlabel("detector pixels")
ax2.set_ylabel("angles")
ax2.set_aspect("auto")

plt.show()


xk = np.zeros_like(phantom)

step_size = 1/(odl.power_method_opnorm(ray_trafo)**2*2)

for i in range(100):
    xk = xk - step_size*ray_trafo.adjoint(ray_trafo(xk) - sinogram)


fig, (ax1, ax2) = plt.subplots(1,2)

ax1.imshow(xk)
ax2.imshow(phantom)

plt.show()