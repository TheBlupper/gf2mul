import sys
import json
import matplotlib.pyplot as plt
from collections import defaultdict

datas = []
for fn in sys.argv[1:]:
    with open(fn) as f:
        datas.extend(json.load(f))

fig, ax = plt.subplots()

mat_szs = sorted(set([data['mat_sz'] for data in datas]))
X = [x for x in mat_szs]

plots = defaultdict(list)
for data in datas:
    plots[data['method_name']].append(data['cycles'])

cmap = plt.get_cmap('hsv', len(plots)+1)
for i, (method_name, cycles) in enumerate(plots.items()):
    Y = [x**3 / cycle for x, cycle in zip(X, cycles)]
    ax.plot(X, Y, label=method_name, color=cmap(i))
ax.legend()

plt.show()