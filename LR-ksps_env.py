#"shufflelunch.py"から学習構築構築

#state:経路の組み合わせ
#action:経路の組み替え
#observation:各辺の負荷率
#reward:最大負荷率に−1をかけたもの

import gym.spaces
import random
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
import copy
import time
import csv
import numpy as np


# class GroupingEnv(gym.core.Env): # クラスの定義
#     #gymで強化学習環境を作る場合はstep,reset,render,close,seedメソッドを実装
    
#     metadata = {'render.modes': ['human', 'rgb_array']}
#     #??????

#     def __init__(self, relationship_matrix, n_group, n_member):
#         self.relationship_matrix = relationship_matrix # 2者間の関係性を表す行列
#         self.n_member = n_member # 全体のメンバー数
#         self.n_group = n_group # グループの数
#         self.grouping = list(range(self.n_group)) * int(self.n_member / self.n_group) # グループ分け
#         self.group_id_list = list(range(n_group)) # groupのidのリスト

#         self.n_action = 10
#         self.action_space = gym.spaces.Discrete(self.n_action) # actionの取りうる値 gym.spaces.Discrete(N):N個の離散値の空間
#         #可能なactionの集合に対してactionにしたがって組み替えたときのコストを計算し、コストの小さい10通りのうち何番目を選ぶか

#         self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(self.n_action,)) # 観測データの取りうる値
#         # n_action通りの組み替えコストを観測データ
        
#         self.time = 0 # ステップ
#         self.max_step = 20 # ステップの最大数

#         self.pair_list = self.get_pair_list()
#         self.candidate_list = [] # 経路の組み替えの候補

#     def step(self, action): # 各ステップで実行される操作
#         #actionの値にしたがってメンバーを入れ替える
#         #⇒メソッドget_observationで入れ替え後の状態に対して観測データを得る
#         #⇒メソッドget_rewardで報酬を計算する
#         #⇒終了条件を満たしているかを判定する

#         self.time += 1
#         # step1: actionに戻づいてグループ分けを更新する
#         self.grouping = self.exchange_path_action(self.grouping, action).copy()
#         # step2: 観測データ(observation)の計算
#         observation = self.get_observation()
#         # step3: 報酬(reward)の計算
#         reward = self.get_reward(self.grouping)
#         # step4: 終了時刻を満たしているかの判定
#         done = self.check_is_done()
#         info = {}
#         return observation, reward, done, info

#     def reset(self): # 変数の初期化。最短経路の組み合わせに初期化する。doneがTrueになったときに呼び出される。
#         self.time = 0
#         self.grouping = self.get_shortest_combination()
#         return self.get_observation()

#     def render(self, mode): # 画面への描画・可視化
#         pass
#     def close(self): # 終了時の処理
#         pass
#     def seed(self): # 乱数の固定
#         pass

#     #####

#     def get_reward(self, grouping): # 報酬を計算する関数
#         return -1 * self.MaxLoadFactor(grouping) # 報酬 = 最大負荷率*(-1)

#     def get_observation(self): # 観測データを計算する関数　経路の組み替えによる負荷率をみたいから、pairでのobservationを確認する
#         grouping = self.grouping.copy()
#         candidate_list = []
#         for pair in self.pair_list:
#             id1, id2 = pair
#             cost = self.get_reward(self.exchange_member_pair(grouping, id1, id2))
#             candidate_list.append([id1, id2, cost])

#         self.candidate_list = sorted(candidate_list, key=lambda x:-x[2])[0:self.n_action]
#         return [cand[2] for cand in self.candidate_list]

#     def check_is_done(self): # 終了条件を判定する関数
#         # 最大数に達したら終了する
#         return self.time == self.max_step
    
#     #####

#     def exchange_path_action(self, grouping, action): # actionの値に応じて、2人のグループを交換する関数　実際に組み替える作業
#         new_grouping = grouping.copy()
#         id1, id2, cost = self.candidate_list[action]
#         new_grouping[id1] = grouping[id2] # id1のメンバーとid2のメンバーを交換
#         new_grouping[id2] = grouping[id1]
#         return new_grouping

#     def exchange_member_pair(self, grouping, id1, id2): # id1, id2のメンバーを交換する関数
#         new_grouping = grouping.copy()
#         new_grouping[id1] = grouping[id2]
#         new_grouping[id2] = grouping[id1]
#         return new_grouping

#     def get_shortest_combination(self): # 最短経路の組み合わせを得るための関数
#         grouping = list(range(self.n_group)) * int(self.n_member / self.n_group)
#         random.shuffle(grouping)
#         return grouping
#         #最短経路の組み合わせになるように変更が必要

#     def MaxLoadFactor(self, grouping): # 最大負荷率を計算する関数
#         # return sum([self.LoadFactor(grouping, group_id) for group_id in self.group_id_list])
#         return max([self.LoadFactor(self, grouping, group_id) for group_id in self.group_id_list])
    
#     def LoadFactor(self, grouping, group_id): # 負荷率の計算
#         # group = [i for i, _x in enumerate(grouping) if _x == group_id]
#         # n_pair = len(group) * (len(group) - 1)
#         # return self.relationship_matrix[np.ix_(group, group)].flatten().sum() / n_pair
#         for e in range(len(g.edges())): #容量制限
#             sum((All_conbination[c][l.get_id()][e])*(l.get_demand())for l in g.all_flows)/capacity[r_kakai[e][1]]
#         return #リストに各辺の負荷率を入れるイメージ
    
#     # def get_pair_list(self): # 可能なペアの列挙
#     #     pair_list = []
#     #     for id1 in list(range(self.n_member)):
#     #         for id2 in list(range(self.n_member)):
#     #             if id1 < id2:
#     #                 pair_list.append([id1, id2])
#     #     return pair_list
#     def get_pair_list(self): # 可能な経路の列挙
#         pair_list = []
#         for id1 in list(range(self.n_member)):
#             for id2 in list(range(self.n_member)):
#                 if id1 < id2:
#                     pair_list.append([id1, id2])
#         return pair_list
    
# #####

def gridgraph(n, m): # gridgraph生成
    G = nx.grid_2d_graph(n, m) 
    n = n*m
    mapping = dict(zip(G, range(1, 100)))
    G = nx.relabel_nodes(G, mapping)
    G = nx.DiGraph(G) # 有向グラフに変換
    G.number_of_area = n
    G.area_nodes_dict = dict()
    G.flow_initialValues = dict()
    G.all_flows = list()
    for a in range(n):
        G.area_nodes_dict[a] = []
    self=G
    for i, j in G.edges(): # capaciy設定
        random.seed()
        v = random.randrange(300, 1000, 100)
        G.adj[i][j]['capacity'] = v
    # nx.write_gml(G,"/Users/takahashihimeno/Documents/result/graph.gml")
    for i, j in G.edges():
        G.add_edge(i, j, flow = dict(self.flow_initialValues), x = dict(), x_kakai = dict())
    pos = nx.spring_layout(G)

    return G, pos

def randomgraph(n, p): # randomgraph生成
    G = nx.fast_gnp_random_graph(n, p)
    while True:
        if nx.is_connected(G) == False: #強連結グラフ判定
            G = nx.fast_gnp_random_graph(n, p)
        else:
            break
        # print("次数", G.degree())
    for i in range(n): # 次数3以上のグラフ作成
        while True:
            if nx.degree(G)[i] < 3:
                j = random.choice(range(n))
                while True:
                    if i != j:
                        G.add_edge(i, j)
                        break
                    else:
                        j = random.choice(range(n))
            else:
                break
    G = nx.DiGraph(G) #有向グラフに変換
    G.number_of_area = n
    G.area_nodes_dict = dict()
    G.flow_initialValues = dict()
    G.all_flows = list()
    for a in range(n):
        G.area_nodes_dict[a] = []
    self = G
    for i, j in G.edges(): #capaciy設定
        random.seed()
        v = random.randrange(300, 1000, 100)
        G.adj[i][j]['capacity'] = v
    # nx.write_gml(G, "/Users/takahashihimeno/Documents/result/graph.gml")
    for i, j in G.edges():
        G.add_edge(i, j, flow = dict(self.flow_initialValues), x = dict(), x_kakai = dict())
    pos = nx.circular_layout(G)

    return G, pos

def show(G, pos): # グラフの参照
    nx.draw_networkx(G, pos, with_labels = True, alpha = 0.5)
    plt.show()

def generate_commodity(commodity): # 品種の定義
    determin_st = []
    commodity_list = []
    for i in range(commodity): # commodity generate
        commodity_dict = {}
        s , t = tuple(random.sample(G.nodes, 2)) # source，sink定義
        demand = random.randint(10, 100) # demand設定
        tentative_st = [s,t] 
        while True:
            if tentative_st in determin_st:
                random.seed()
                s , t = tuple(random.sample(G.nodes, 2)) # source，sink再定義
                tentative_st = [s,t]
            else:
                break 
        determin_st.append(tentative_st) # commodity決定
        commodity_dict["id"] = i
        commodity_dict["source"] = s 
        commodity_dict["sink"] = t
        commodity_dict["demand"] = demand
        # print("commodity_list",commodity_list)
        commodity_list.append(commodity_dict)
        # print("commodity_dict",commodity_dict)
        # print("commodity_list",commodity_list)

    # with open('/Users/takahashihimeno/Documents/result/variety.csv','w') as f:
    #     writer=csv.writer(f,lineterminator='\n')
    #     writer.writerows(variety)

    return commodity_list

def search_ksps(K,G,commodity,commodity_list): # 各品種のkspsを求める
    allcommodity_ksps = []
    for i in range(commodity):
        X = nx.shortest_simple_paths(G, commodity_list[i]['source'], commodity_list[i]['sink']) # Yen's algorithm
        ksps_list = [] 
        for counter, path in enumerate(X):            
            ksps_list.append(path) 
            if counter == K - 1: 
                break
            allcommodity_ksps.append(ksps_list)
    return allcommodity_ksps

def searh_combination(allcommodity_ksps): # 組み合わせを求める
    combination = []
    q = [*product(*allcommodity_ksps)] # 全通りの組み合わせ
    for conbi in q:
        combination.append(conbi)
    return combination

print("--------------------------------------------------")
print("")

# # 組み分けの環境の定義
# n_member = 20 # 全体のメンバー数
# n_group = 4 #グループ数
# n_action = 10

# グラフの定義
retsu = 2 # 行列
node = 10 #　ノード数
p = 0.15 #　リンクを張る密度
G, pos = gridgraph(retsu, retsu) # gridgraph
# G,pos = randomgraph(node, p) #　randomgraph
# show(G,pos)

# 品種の定義
commodity = 2 # 品種数
commodity_list = generate_commodity(commodity)
print("commodity_list:",commodity_list)
# print(commodity_list[0]['sink'])

# KSPsの探索
K = 2 # パスの個数
allcommodity_ksps = search_ksps(K,G,commodity,commodity_list)
print("allcommodity_ksps:",allcommodity_ksps)

# 組み合わせの探索
combination = searh_combination(allcommodity_ksps)
print("conbination:",combination)
print("The number of combination:",len(combination))
print("")
print("--------------------------------------------------")

# # 親密度行列の定義
# # NOTE: ランダム行列で代用
# relationship_matrix = np.array([random.random() for _ in range(n_member ** 2)]).reshape(n_member, n_member)
# env = GroupingEnv(relationship_matrix, n_group, n_member)