# -*- coding: utf-8 -*-
"""
百度地图路径规划工具 - 最短时间路径优化版
功能：
1. 计算所有点位间的行驶时间
2. 找出时间最短的路径组合
3. 支持可视化展示
"""

import requests
import json
import pandas as pd
import time
import folium
import webbrowser
from itertools import permutations
# what can i say
# 俺不会写

AK = "L7y9a6CoWnudPHlsmAhGQzp47TSoVBQJ"

# 硬编码的坐标数据
POINTS = {
    31: (111.042637, 38.112507),
    44: (111.073839, 38.117061),
    61: (111.069797, 38.116762),
    83: (111.057493, 38.124064),
    100: (111.044954, 38.119702),
    115: (111.040331, 38.110755),
    147: (111.046929, 38.118705),
    158: (111.065254, 38.127754)
}

def get_driving_time(start_lng, start_lat, end_lng, end_lat):
    """获取两点间的驾车时间(秒)"""
    url = f"https://api.map.baidu.com/directionlite/v1/driving?origin={start_lat},{start_lng}&destination={end_lat},{end_lng}&ak={AK}"
    
    try:
        time.sleep(0.5)  # API限流
        res = requests.get(url, timeout=10)
        data = res.json()
        
        if data.get("status") == 0:
            return data["result"]["routes"][0]["duration"]
        return float('inf')  # 无法计算时返回无限大
    except:
        return float('inf')

def build_time_matrix():
    """构建所有点位间的时间矩阵"""
    print("正在计算时间矩阵...")
    ids = list(POINTS.keys())
    time_matrix = {}
    
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            start_id = ids[i]
            end_id = ids[j]
            
            # 获取行驶时间
            duration = get_driving_time(
                POINTS[start_id][0], POINTS[start_id][1],
                POINTS[end_id][0], POINTS[end_id][1]
            )
            
            # 双向时间相同（假设）
            time_matrix[(start_id, end_id)] = duration
            time_matrix[(end_id, start_id)] = duration
            
            print(f"{start_id} → {end_id}: {duration//60}分{duration%60}秒")
    
    # 对角线设为0（同一点）
    for id in ids:
        time_matrix[(id, id)] = 0
        
    return time_matrix

def find_shortest_path(time_matrix, start_id=None):
    """寻找访问所有点的最短时间路径"""
    point_ids = list(POINTS.keys())
    
    if start_id:
        point_ids.remove(start_id)
        point_ids.insert(0, start_id)  # 固定起点
    
    # 生成所有可能路径排列（限制点数避免组合爆炸）
    if len(point_ids) > 8:
        print("警告：点数过多，使用近似算法")
        return approximate_shortest_path(time_matrix, point_ids)
    else:
        print("正在计算所有可能路径...")
        return exact_shortest_path(time_matrix, point_ids)

def exact_shortest_path(time_matrix, point_ids):
    """精确算法（点数≤7时使用）"""
    min_time = float('inf')
    best_path = None
    
    for path in permutations(point_ids[1:]):
        current_path = [point_ids[0]] + list(path)
        total_time = 0
        
        # 计算路径总时间
        for i in range(len(current_path)-1):
            total_time += time_matrix[(current_path[i], current_path[i+1])]
        
        if total_time < min_time:
            min_time = total_time
            best_path = current_path
    
    return best_path, min_time

def approximate_shortest_path(time_matrix, point_ids):
    """近似算法（最近邻启发式）"""
    unvisited = set(point_ids)
    path = [point_ids[0]]
    unvisited.remove(point_ids[0])
    total_time = 0
    
    while unvisited:
        last = path[-1]
        nearest = min(unvisited, key=lambda x: time_matrix[(last, x)])
        total_time += time_matrix[(last, nearest)]
        path.append(nearest)
        unvisited.remove(nearest)
    
    return path, total_time

def visualize_path(path):
    """可视化最短路径"""
    if not path or len(path) < 2:
        print("无效路径")
        return
    
    # 创建地图
    first_point = POINTS[path[0]]
    m = folium.Map(location=[first_point[1], first_point[0]], zoom_start=13)
    
    # 添加所有路径段
    for i in range(len(path)-1):
        start = POINTS[path[i]]
        end = POINTS[path[i+1]]
        
        # 获取路线细节
        route = get_route_details(start[0], start[1], end[0], end[1])
        if route and route['path']:
            folium.PolyLine(
                locations=[(lat, lng) for lng, lat in route['path']],
                color='blue',
                weight=3,
                popup=f"{path[i]}→{path[i+1]}"
            ).add_to(m)
        
        # 添加标记
        folium.Marker(
            location=[start[1], start[0]],
            popup=f"点位{path[i]}",
            icon=folium.Icon(color='green' if i==0 else 'blue')
        ).add_to(m)
    
    # 添加终点标记
    last = POINTS[path[-1]]
    folium.Marker(
        location=[last[1], last[0]],
        popup=f"点位{path[-1]}",
        icon=folium.Icon(color='red')
    ).add_to(m)
    
    # 保存并打开地图
    map_file = "shortest_path.html"
    m.save(map_file)
    webbrowser.open(map_file)

def get_route_details(start_lng, start_lat, end_lng, end_lat):
    """获取路线详细信息"""
    url = f"https://api.map.baidu.com/directionlite/v1/driving?origin={start_lat},{start_lng}&destination={end_lat},{end_lng}&ak={AK}"
    
    try:
        time.sleep(0.5)
        res = requests.get(url, timeout=10)
        data = res.json()
        
        if data.get("status") == 0:
            route = data["result"]["routes"][0]
            path = []
            
            for step in route["steps"]:
                points = step["path"].split(';')
                for point in points:
                    lng, lat = map(float, point.split(','))
                    path.append((lng, lat))
            
            return {
                'distance': route["distance"],
                'duration': route["duration"],
                'path': path
            }
        return None
    except:
        return None

def save_results(path, total_time, time_matrix):
    """保存结果到Excel"""
    if not path:
        print("无有效路径")
        return
    
    # 准备数据
    data = []
    for i in range(len(path)-1):
        start = path[i]
        end = path[i+1]
        duration = time_matrix[(start, end)]
        
        data.append([
            f"点位{start}", f"点位{end}",
            POINTS[start][0], POINTS[start][1],
            POINTS[end][0], POINTS[end][1],
            duration//60, duration%60,
            f"{duration//60}分{duration%60}秒"
        ])
    
    # 创建DataFrame
    df = pd.DataFrame(data, columns=[
        '起点', '终点',
        '起点经度', '起点纬度',
        '终点经度', '终点纬度',
        '分钟数', '秒数',
        '行驶时间'
    ])
    
    # 保存文件
    excel_file = "shortest_path_results.xlsx"
    df.to_excel(excel_file, index=False)
    print(f"\n结果已保存到: {excel_file}")
    
    # 打印汇总信息
    print(f"\n===== 最短路径方案 =====")
    print(f"总点数: {len(path)}")
    print(f"总行驶时间: {total_time//60}分{total_time%60}秒")
    print("\n路径顺序:")
    print(" → ".join(f"点位{id}" for id in path))

if __name__ == "__main__":
    print("===== 最短时间路径规划工具 =====")
    print(f"共有 {len(POINTS)} 个点位")
    
    # 1. 构建时间矩阵
    time_matrix = build_time_matrix()
    
    # 2. 寻找最短路径
    start_point = int(input("\n请输入起点编号(可选，直接回车跳过): ") or 0)
    if start_point in POINTS:
        path, total_time = find_shortest_path(time_matrix, start_point)
    else:
        path, total_time = find_shortest_path(time_matrix)
    
    # 3. 结果处理
    if path:
        save_results(path, total_time, time_matrix)
        visualize_path(path)
    else:
        print("未能找到有效路径")
        