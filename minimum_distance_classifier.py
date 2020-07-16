import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train.txt', sep=' ', header=None, dtype='Int64')
df_arr = df.values
l = df_arr[:, 0].size
train_max = np.max(df_arr)
train_min = np.min(df_arr)

class_1 = []
class_2 = []
for i in range(l):
    if df.loc[i, 2] == 1:
        class_1.extend([df_arr[i, 0:2]])
    elif df.loc[i, 2] == 2:
        class_2.extend([df_arr[i, 0:2]])

class_1 = np.array(class_1)
class_2 = np.array(class_2)
x1 = class_1[:, 0]
y1 = class_1[:, 1]
x2 = class_2[:, 0]
y2 = class_2[:, 1]

plt.scatter(x1, y1, color='red', marker='o')
plt.scatter(x2, y2, color='green', marker='p')
#plt.show()

mean_1 = np.array([np.mean(x1), np.mean(y1)])
mean_2 = np.array([np.mean(x2), np.mean(y2)])

dtest = pd.read_csv('test.txt', sep=' ', header=None, dtype='Int64')
dtest_arr = dtest.values
test_max = np.max(dtest_arr)
test_min = np.min(dtest_arr)


tl = dtest_arr[:, 0].size
class_1 = []
class_2 = []
d = []


def mindist(x, mn):
    return np.dot(x, mn) - 0.5 * np.dot(mn, mn)


for i in range(tl):
    dist_1 = mindist(dtest_arr[i, 0:2], mean_1)
    dist_2 = mindist(dtest_arr[i, 0:2], mean_2)
    if dist_1 <= dist_2:
        dtest_arr[i][2] = 1
        class_1.extend([dtest_arr[i]])
        d.extend([dtest_arr[i]])
    else:
        dtest_arr[i][2] = 2
        class_2.extend([dtest_arr[i]])
        d.extend([dtest_arr[i]])

final = []
c_1 = []
c_2 = []


final = np.array(d)
c_1 = np.array(class_1)
c_2 = np.array(class_2)

dtest = pd.read_csv('test.txt', sep=' ', header=None, dtype='Int64')
dtest_arr = dtest.values

x1 = c_1[:, 0]
y1 = c_1[:, 1]
x2 = c_2[:, 0]
y2 = c_2[:, 1]

plt.scatter(x1, y1, color='red', marker='+')
plt.scatter(x2, y2, color='green', marker='*')


temp = 0

for i in range(tl):
    if final[i][2] == dtest_arr[i][2]:
        temp += 1
acc = (temp/tl)*100

print(acc, "% accurate")


limit_max = max(train_max, test_max)
limit_min = min(train_min, test_min)

b = np.linspace(limit_min, limit_max, 100)

m1 = []


def boundary(mean_1, mean_2, x):
    c = .5*(np.dot(mean_1,mean_1) - np.dot(mean_2, mean_2))
    m1 = mean_1 - mean_2
    ans = (c - m1[0]*x)/m1[1]
    return ans


line = np.zeros(len(b))

for i in range(len(b)):
    line[i] = boundary(mean_1, mean_2, b[i])

plt.plot(b, line, '--')
plt.show()