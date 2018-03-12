import numpy as np
from matplotlib import pyplot as plt

vs_1 = np.array([[2,2], [3,3], [4,4]])
s_best = np.array([5,5])
i_best = np.array([[5,5], [7,3], [5,6]])
pos_1 = np.array([[5,5], [8,3], [6,7]])


update = lambda w, v, sb, ib, p, r: w*v + r[0]*(sb-p) + r[1]*(ib-p)

r = [.5,.5]
w = 2
vs_2 = update(w,vs_1,np.tile(s_best,(len(vs_1),1)),i_best,pos_1,r)
pos_2 = pos_1 + vs_2

print(pos_2)

w = .1
vs_2 = update(w,vs_1,np.tile(s_best,(len(vs_1),1)),i_best,pos_1,r)
pos_2 = pos_1 + vs_2

print(pos_2)



for w in np.arange(0,1,0.1):
	v = .5
	b = 3
	p = 10
	iter = 500
	pos = np.zeros(iter)
	for i in range(iter):
		pos[i] = p
		v = update(w,v,b,b,p,[0,np.random.rand()])
		p += v
		if 1**2 < b**2:
			b = p
	plt.plot(pos)

plt.show()