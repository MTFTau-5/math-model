import itertools
import math
import matplotlib.pyplot as plt
import networkx as nx

#计算两点间的平面距离
#因为经纬度相差较小，这里我选择了近似为平地计算距离（我都这么干的）
#也就是直接欧式距离计算
#如果需要更精确的计算，可以使用haversine公式（后面的优化直接用这个就行，毕竟有误差）
points = {
    31: (111.042637, 38.112507),
    44: (111.073839, 38.117061),
    61: (111.069797, 38.116762),
    83: (111.057493, 38.124064),
    100: (111.044954, 38.119702),
    115: (111.040331, 38.110755),
    147: (111.046929, 38.118705),
    158: (111.065254, 38.127754)
}

def flat_distance(lon1, lat1, lon2, lat2):
    dlon = (lon2 - lon1) * 87.5  
    dlat = (lat2 - lat1) * 111   
    return math.sqrt(dlon**2 + dlat**2)
dist = {}
keys = list(points.keys())
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        d = flat_distance(
            points[keys[i]][0], points[keys[i]][1],
            points[keys[j]][0], points[keys[j]][1]
        )
        dist[(keys[i], keys[j])] = d
        dist[(keys[j], keys[i])] = d

min_length = float('inf')
best_path = []
for perm in itertools.permutations(keys):
    current_length = sum(dist[(perm[i], perm[i+1])] for i in range(len(perm)-1))
    if current_length < min_length:
        min_length = current_length
        best_path = perm

print("最短路径顺序:", best_path)
print("总距离（km）:", round(min_length, 3))


G = nx.DiGraph()
for i in range(len(best_path)-1):
    u, v = best_path[i], best_path[i+1]
    G.add_edge(u, v, weight=dist[(u, v)])

lons = [points[node][0] for node in G.nodes()]
lats = [points[node][1] for node in G.nodes()]
plt.figure(figsize=(10, 8))
plt.scatter(lons, lats, c='red', s=100, zorder=2)  
for i, node in enumerate(G.nodes()):
    plt.text(lons[i], lats[i], f"{node}", fontsize=12, ha='right', va='bottom') 
path_edges = list(zip(best_path[:-1], best_path[1:]))
for u, v in path_edges:
    plt.plot(
        [points[u][0], points[v][0]],
        [points[u][1], points[v][1]],
        'b-', linewidth=2, zorder=1
    )

plt.title("最短路径可视化", fontsize=14)
plt.xlabel("经度", fontsize=12)
plt.ylabel("纬度", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()