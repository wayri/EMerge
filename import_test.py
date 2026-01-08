import numpy as np

with open('test.txt', 'r') as f:
    content = f.read()

theta, phi, *data = content.split('\n\n\n')
theta = [float(x) for x in theta.split(',')]
phi = [float(x) for x in phi.split(',')]
print(np.array(theta).shape)
print(np.array(phi).shape)

for d in data:
    f, Ex, Ey, Ez = d.split('\n\n')
    print(f)
    
    Ex_arry = np.array([[complex(x) for x in line.split(',')] for line in Ex.split('\n')])
    Ey_arry = np.array([[complex(x) for x in line.split(',')] for line in Ey.split('\n')])
    Ez_arry = np.array([[complex(x) for x in line.split(',')] for line in Ez.split('\n')])
    print(Ex_arry.shape, Ey_arry.shape, Ez_arry.shape)