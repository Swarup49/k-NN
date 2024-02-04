import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# feature set containing (x,y) values of 25 known/training data
train_data = np.random.randint(0,100,(25,2)).astype(np.float32)
print(train_data)

# Label each either Red or Blue with numbers 0 and 1
responses = np.random.randint(0,2,(25,1)).astype(np.float32)
print(responses)

# Red neighbour plotting
red = train_data[responses.ravel()==0]
plt.scatter(red[:,0], red[:,1], 80, 'r', '^')

# Blue neighbours plotting
blue = train_data[responses.ravel()==1]
plt.scatter(blue[:,0], blue[:,1], 80,'b','s')

"""NExt i will initiate the kNN algorithm and pass the tranData and responses to train the kNN"""
newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
plt.scatter(newcomer[:,0], newcomer[:,1], 80, 'g', 'o')

plt.savefig('scatter_plot.jpg')

knn = cv.ml.KNearest_create()
knn.train(train_data, cv.ml.ROW_SAMPLE, responses)
ret, results, neighbours, dist = knn.findNearest(newcomer, 3)

print('result: {}\n'.format(results))
print('neighbours: {}\n'.format(neighbours))
print('distance: {}\n'.format(dist))

plt.show()


