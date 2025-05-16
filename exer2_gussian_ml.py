import numpy as np;
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.linalg as la


#Load the dataset 
data = np.load('data/gaussian.npz')
points = data['arr_0']

#step1: compute mean (MLE)
mean = np.mean(points, axis=0)

#step2: compute the covariance matrix(MLE)
cov = np.cov(points.T, bias=True)

#step3: eigen decompostition of covariance matrix
eigenvalues, eigenvectors = la.eigh(cov)


#step4: prepare ellipses parameters
#sort eigenvalues and eigenvectors in descending orders
order = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

#Scale for 1 standard deviation ellipse (68% points)
scale = 1.0
width = 2 * np.sqrt(eigenvalues[0]) * scale # major axis
height = 2 * np.sqrt(eigenvalues[1]) * scale # minor axis

#Compute the angle of major axis (in degrees)
angle = np.arctan2(eigenvectors[0,1], eigenvectors[0,0]) * 180 / np.pi

#step5: plotting
fig, ax = plt.subplots()

#plot data points
ax.scatter(points[:, 0], points[:, -1], alpha = 0.5, label = 'Data points')

#plot mean as cross
ax.plot(mean[0], mean[1], 'rx', markersize = 15, label='Mean')

#Create and add the covariance ellipse
ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='red', fc='none', lw=2, label='Covariance ellipse')
ax.add_patch(ellipse)

#customize the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Gussian distribution with mean and covariance ellipse')
ax.legend()
ax.grid(True)
ax.set_aspect('equal')

#show the plot
plt.show()





