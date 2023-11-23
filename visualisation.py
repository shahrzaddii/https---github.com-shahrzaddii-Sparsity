import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import pandas as pd



octet = pd.read_csv('octet.csv')
octet_vis = octet[['Delta E', 'rp(A)', 'rp(B)', 'rs(A)']].copy()
octet_vis.loc[:, 'sigma'] = abs(octet_vis['rp(A)']-octet_vis['rp(B)'])
octet_vis.loc[(octet_vis['sigma']>=0.91) & (octet_vis['rs(A)'] >=1.22), 'RS_ZB'] = 1 
octet_vis.loc[(octet_vis['sigma']<=1.16) & (octet_vis['rs(A)'] <=1.27), 'RS_ZB'] = -1 
octet_vis.loc[octet_vis['RS_ZB'].isnull(), 'RS_ZB'] = 0

fig = plt.figure(figsize=(7,7))
ax = plt.axes(projection='3d')
zdata = octet_vis['rs(A)']
ydata = octet_vis['rp(A)']
xdata = octet_vis['rp(B)']
# ax.scatter3D(xdata, ydata, zdata, c=octet_vis['RS_ZB'], cmap='viridis')
ax.set_xlabel('rp(A)')
ax.set_ylabel('rp(B)')
ax.set_zlabel('rs(A)')
# # ax.set_xticks([])
# # ax.set_yticks([])
# ax.set_zticks([])

x_rpB_1 = np.linspace(0,1, 100)
y_rpA_1 = x_rpB_1 + 0.91
z_1 = np.linspace(0,3, 100)
y_rpA_1, z_1 = np.meshgrid(y_rpA_1, z_1)
ax.plot_surface(x_rpB_1, y_rpA_1, z_1, alpha=0.3, color='blue')
ax.plot(x_rpB_1,y_rpA_1)

x_rpB_2 = np.linspace(min(xdata),max(xdata), 100)
y_rpA_2 = x_rpB_2 - 0.91
z_2 = np.linspace(0,3, 100)
x_rpB_2, z_2 = np.meshgrid(x_rpB_2, z_2)
ax.plot_surface(x_rpB_2, y_rpA_2, z_2, alpha=0.3, color='blue')


x_rpB_3 = np.linspace(min(xdata),max(xdata), 100)
y_rpA_3 = x_rpB_3 + 1.16
z_3 = np.linspace(0,3, 100)
x_rpB_3, z_3 = np.meshgrid(x_rpB_3, z_3)
ax.plot_surface(x_rpB_3, y_rpA_3, z_3, alpha=0.3, color='green')

x_rpB_4 = np.linspace(min(xdata),max(xdata), 100)
y_rpA_4 = x_rpB_4 - 1.16
z_4 = np.linspace(0,3, 100)
x_rpB_4, z_4 = np.meshgrid(x_rpB_4, z_4)
ax.plot_surface(x_rpB_4, y_rpA_4, z_4, alpha=0.3, color='green')

x_rpB_5 = np.linspace(min(xdata),max(xdata), 100)
y_rpA_5 = np.linspace(min(ydata),max(ydata), 100)
z_5 = np.array([1.22])
x_rpB_5, y_rpA_5 = np.meshgrid(x_rpB_5, y_rpA_5)
ax.plot_surface(x_rpB_5, y_rpA_5, z_5, alpha=0.3, color='red')


plt.show()
