import numpy as np

m = np.random.randint(0,100, size=(10,20,20))

sumtmp = 0
for i in range(10):
    sumtmp += m[i][1][1]

print(sumtmp/10)
print(np.mean(m, axis=0)[1][1])

