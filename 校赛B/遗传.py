
import argparse
import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib import font_manager
import random
from tqdm import tqdm
import os
# 这里我选择了一个另辟蹊径
# 也就是计算闭合的，然后除去最长的一条边
# 这样就是最长的就是最优路径
# 当然也可以直接计算最短路径
# 但是这样就需要计算所有的路径了
# 遗传算法写起来比较费劲
# 所以我选择了这种方法
# 但是这种方法也有一些问题
# 就是可能会出现三角形不等式
# 也就是两条边的和大于第三条边
# 在减去最长边的时候可能前一个边取得并不是最优的
# 倒数第3个节点取得是较远的点
# 但是倒数第2个节点取得是较近最后的点
# 这样能保证整体最短
# 但是我觉得这个问题不大
# 因为三角形不等式是一个近似值（数值不会相差到极大）
# 而且我都用遗传算法了
# 本身他就是一个找近似值的算法
# 所以我觉得这个问题不大
parser = argparse.ArgumentParser(description='遗传算法配置')
arg_lists = []

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

data_arg = add_argument_group('Algorithm')
data_arg.add_argument('--city_num', type=int, default=222, help='城市数量')
data_arg.add_argument('--individual_num', type=int, default=150, help='种群规模')
data_arg.add_argument('--gen_num', type=int, default=10000, help='迭代次数')
data_arg.add_argument('--mutate_prob', type=float, default=0.25, help='基础变异概率')
data_arg.add_argument('--elite_size', type=int, default=4, help='保留精英数量')
data_arg.add_argument('--elite_protect', type=float, default=0.7, help='精英保护概率')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config

class Individual:
    def __init__(self, genes=None):
        if genes is None:
            genes = list(range(config.city_num))
            random.shuffle(genes)
        self.genes = genes
        self.fitness = self.evaluate_fitness()
        
    def evaluate_fitness(self):
        total = 0.0
        for i in range(len(self.genes)-1):
            total += city_dist_mat[self.genes[i], self.genes[i+1]]
        return total
    
    def diversity(self, population):
        avg_diff = sum(len(set(self.genes) - set(ind.genes)) for ind in population)
        return avg_diff / (len(population) * len(self.genes))

class GeneticAlgorithm:
    def __init__(self, distance_matrix):
        global city_dist_mat
        city_dist_mat = distance_matrix
        self.best = None
        self.individual_list = []
        self.fitness_history = []
        self.elite_history = []
        
    def initialize_population(self):
        self.individual_list = [Individual() for _ in range(config.individual_num)]
        self.best = min(self.individual_list, key=lambda x: x.fitness)
        
    def tournament_selection(self):
        winners = []
        for _ in range(config.individual_num - config.elite_size):
            competitors = random.sample(self.individual_list, 
                                      min(10, len(self.individual_list)))
            winner = min(competitors, key=lambda x: x.fitness)
            winners.append(Individual(winner.genes.copy()))
        return winners
    
    def order_crossover(self, parent1, parent2):
        size = len(parent1.genes)
        idx1, idx2 = sorted(random.sample(range(size), 2))
        

        child1 = [-1] * size
        child2 = [-1] * size
        child1[idx1:idx2] = parent1.genes[idx1:idx2]
        child2[idx1:idx2] = parent2.genes[idx1:idx2]
        for child, parent in [(child1, parent2), (child2, parent1)]:
            ptr = idx2
            for gene in parent.genes[idx2:] + parent.genes[:idx2]:
                if gene not in child[idx1:idx2]:
                    if ptr >= size:
                        ptr = 0
                    child[ptr] = gene
                    ptr += 1
                    
        return Individual(child1), Individual(child2)
    
    def mutate(self, individual):
        if random.random() < config.mutate_prob:
            idx1, idx2 = sorted(random.sample(range(len(individual.genes)), 2))
            individual.genes[idx1:idx2] = reversed(individual.genes[idx1:idx2])
            individual.fitness = individual.evaluate_fitness()
    
    def protect_elites(self):
        elites = sorted(self.individual_list, key=lambda x: x.fitness)[:config.elite_size]
        protected = []
        for elite in elites:
            if random.random() < config.elite_protect:
                protected.append(Individual(elite.genes.copy()))
        return protected
    
    def next_generation(self):
        # 精英保留
        elites = self.protect_elites()
        winners = self.tournament_selection()
        offspring = []
        for i in range(0, len(winners)-1, 2):
            child1, child2 = self.order_crossover(winners[i], winners[i+1])
            offspring.extend([child1, child2])
        
        for ind in offspring:
            self.mutate(ind)
            

        self.individual_list = elites + offspring[:config.individual_num - len(elites)]
        current_best = min(self.individual_list, key=lambda x: x.fitness)
        if self.best is None or current_best.fitness < self.best.fitness:
            self.best = current_best
    
    def train(self):
        self.initialize_population()
        
        with tqdm(total=config.gen_num, desc="优化进度") as pbar:
            for gen in range(config.gen_num):
                self.next_generation()
                if gen % 10 == 0:
                    diversity = self.best.diversity(self.individual_list)
                    pbar.set_postfix
                self.fitness_history.append(self.best.fitness)
                elites = sorted(self.individual_list, key=lambda x: x.fitness)[:config.elite_size]
                self.elite_history.append([e.fitness for e in elites])
                pbar.update(1)
                if gen % 100 == 0:
                    pbar.set_postfix({
                        '最优': f"{self.best.fitness:.2f}h",
                        '精英平均': f"{np.mean([e.fitness for e in elites]):.2f}h"
                    })
                    
        return self.best.genes, self.fitness_history
def load_data(filepath):
    """加载并预处理数据"""
    df = pd.read_excel(filepath)
    data = df[['JD', 'WD', '完成工作所需时间（分钟）']].values
    lon_scale = 111 
    lat_scale = 111  
    coordinates = np.column_stack([
        (data[:,0] - data[0,0]) * lon_scale,
        (data[:,1] - data[0,1]) * lat_scale
    ])
    
    dist_matrix = distance.cdist(coordinates, coordinates, 'euclidean')
    time_matrix = dist_matrix / 20  
    
    return coordinates, time_matrix

def set_chinese_font():
    """设置中文字体"""
    try:
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', 
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  
            '/usr/share/fonts/truetype/arphic/uming.ttc'  
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                font_prop = font_manager.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
                plt.rcParams['axes.unicode_minus'] = False
                break
    except:
        print("警告: 中文字体设置失败，将使用默认字体")

def save_segment_times(path, time_matrix, filename='segment_times.csv'):
    """保存每段路径的时间数据"""
    segment_data = []
    for i in range(len(path)-1):
        from_node = path[i]
        to_node = path[i+1]
        time = time_matrix[from_node, to_node]
        segment_data.append([f"{from_node+1}→{to_node+1}", time])
    
    total_time = sum(time for _, time in segment_data)
    segment_data.append(["总时间", total_time])
    
    df = pd.DataFrame(segment_data, columns=['路段', '时间(小时)'])
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"已保存路段时间数据到 {filename}")

def visualize_results(coords, best_path, fitness_history):
    set_chinese_font()  # 设置中文字体
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    path_coords = coords[best_path]
    plt.plot(path_coords[:,0], path_coords[:,1], 'o-', markersize=4)
    plt.scatter(*path_coords[0], c='red', s=100, label='起点')
    plt.scatter(*path_coords[-1], c='green', s=100, label='终点')
    
    # 标注关键点
    for i, (x, y) in enumerate(path_coords):
        if i % 10 == 0 or i in [0, len(path_coords)-1]:
            plt.text(x, y, str(i+1), fontsize=8, ha='center', va='bottom')
    
    plt.title(f"最优路径规划\n总时间: {fitness_history[-1]:.2f}小时")
    plt.legend()
    
    # 收敛曲线
    plt.subplot(1, 2, 2)
    plt.plot(fitness_history, label='种群最优')
    if hasattr(config, 'elite_size') and config.elite_size > 0:
        elite_avg = [np.mean(x) for x in ga.elite_history]
        plt.plot(elite_avg, label='精英平均')
    plt.xlabel('迭代次数')
    plt.ylabel('总时间(小时)')
    plt.title('适应度收敛曲线')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    config = get_config()
    
  
    print("正在加载数据...")
    coordinates, time_matrix = load_data('/home/mtftau-5/workplace/25中北校赛/2025B土壤普查/附件1：xx地区.xlsx')
    
    print("开始优化...")
    ga = GeneticAlgorithm(time_matrix)
    best_path, fitness_history = ga.train()
    
    def remove_longest_segment(path):
        max_len, cut_idx = -1, 0
        for i in range(len(path)-1):
            seg_len = time_matrix[path[i], path[i+1]]
            if seg_len > max_len:
                max_len, cut_idx = seg_len, i+1
        return path[:cut_idx] + path[cut_idx:]
    
    open_path = remove_longest_segment(best_path)
    print(f"\n最优路径总时间: {fitness_history[-1]:.2f}小时")
    print(f"起点ID: {open_path[0]+1}, 终点ID: {open_path[-1]+1}")
    

    save_segment_times(open_path, time_matrix)
    
    print("生成可视化结果...")
    visualize_results(coordinates, open_path, fitness_history)