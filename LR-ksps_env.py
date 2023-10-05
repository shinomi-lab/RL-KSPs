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


class min_maxload_KSPs_Env(gym.core.Env): # クラスの定義
    #gymで強化学習環境を作る場合はstep,reset,render,close,seedメソッドを実装

    def __init__(self, K, n_action, obs_low, obs_high, max_step, grid_node, range_commodity):
    # def __init__(self, G, capacity_list, edge_list, edges_notindex, commodity, commodity_list, allcommodity_ksps, allcommodity_notfstksps, combination):
    # def __init__(self, G, capacity_list, edge_list, edges_notindex, commodity, commodity_list, allcommodity_ksps, allcommodity_notfstksps, combination, zo_combination):
        self.K = K # kspsの本数
        self.retsu = random.randint(2, grid_node)
        self.commodity = random.randint(2, range_commodity)
        # self.commodity = commodity # 品種数
        # self.G = G # グラフ
        # self.capacity_list = capacity_list # リンク容量のリスト
        # self.edge_list = edge_list # リンクのリスト
        # self.edges_notindex = edges_notindex
        # self.commodity_list = commodity_list # 品種のリスト
        # self.allcommodity_ksps = allcommodity_ksps # KSPsのリスト　pairlistと一緒？？？
        # self.allcommodity_notfstksps = allcommodity_notfstksps
        # self.combination = combination # 組み合わせのリスト
        # self.grouping = combination[0] #初期パターン（最短経路の組み合わせ）

        self.G, self.pos, self.capacity, self.edge_list, self.edges_notindex =  self.gridgraph(self.retsu,self.retsu) # グラフ
        self.commodity_dictlist, self.commodity_list = self.generate_commodity(self.commodity) # 品種作成
        self.allcommodity_ksps, self.allcommodity_notfstksps = self.search_ksps(self.K, self.G, self.commodity, self.commodity_list) # kspsの探索
        self.combination = self.searh_combination(self.allcommodity_ksps)
        self.grouping = self.combination[0] #初期パターン（最短経路の組み合わせ）

        # self.zo_combination = zo_combination
        # self.combination_id = list(len(combination))

        self.n_action = n_action # 行動の数
        self.action_space = gym.spaces.Discrete(self.n_action) # actionの取りうる値 gym.spaces.Discrete(N):N個の離散値の空間
        #可能なactionの集合に対してactionにしたがって組み替えたときのコストを計算し、コストの小さい10通りのうち何番目を選ぶか

        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, shape=(self.n_action,)) # 観測データの取りうる値
        # n_action通りの組み替えコストを観測データ
        
        self.time = 0 # ステップ
        self.max_step = max_step # ステップの最大数

        self.candidate_list = [] # 経路の組み替えの候補

    def render(self, mode): # 画面への描画・可視化
        pass
    def close(self): # 終了時の処理
        pass
    def seed(self): # 乱数の固定
        pass
    def check_is_done(self): # 終了条件を判定する関数
        # 現在：max_stepに到達したら終了
        # 他の方法1：目的関数に達したら終了　▷ solverを使用して目的関数を求める部分が必要
        # 他の方法2：解が余り変わらなくなったら終了
        print("self.time_in_check_is_done",self.time)
        return self.time == self.max_step # 最大数に達したら終了する

    def get_pair_list(self): # 可能な品種と経路のペアを列挙
        pair_list = []
        nothingfstpath_pair_list = []
        for c in range(self.commodity):
            for p in range(len(self.allcommodity_ksps[c])):
                path = self.allcommodity_ksps[c][p]
                pair_list.append([c, path])
                nothingfstpath_pair_list.append([c, path])
                if p == 0:
                    nothingfstpath_pair_list.pop()
        # print("pair_list",pair_list)
        # print("nothingfstpath_pair_list",nothingfstpath_pair_list)
        return pair_list,nothingfstpath_pair_list

   ##### resetで必要な関数 #####
    def get_random_grouping(self): #本番用
        retsu = random.randint(2, grid_node)
        self.commodity = random.randint(2, range_commodity)

        self.G, self.pos, self.capacity_list, self.edge_list, self.edges_notindex = self.gridgraph(retsu,retsu) # グラフ作成
        self.commodity_dictlist, self.commodity_list = self.generate_commodity(self.commodity) # 品種作成
        self.allcommodity_ksps, self.allcommodity_notfstksps = self.search_ksps(self.K, self.G, self.commodity, self.commodity_list) # kspsの探索
        self.combination = self.searh_combination(self.allcommodity_ksps)
        self.grouping = self.combination[0]

        return self.grouping
    
    def get_random_grouping_debug(self): # デバッグ用
        retsu = random.randint(2, grid_node)
        self.commodity = random.randint(2, range_commodity)
        
        self.commodity = 2
        self.G = self.debaggraph(2,2)
        self.capacity_list = {(1, 3): 600, (1, 2): 700, (2, 1): 700, (2, 4): 400, (3, 1): 700, (3, 4): 800, (4, 2): 800, (4, 3): 900}
        self.edge_list = [(0, (1, 3)), (1, (1, 2)), (2, (2, 1)), (3, (2, 4)), (4, (3, 1)), (5, (3, 4)), (6, (4, 2)), (7, (4, 3))]
        self.edges_notindex = [(1, 3), (1, 2), (2, 1), (2, 4), (3, 1), (3, 4), (4, 2), (4, 3)]
        self.commodity_list = [[2, 3, 87], [1, 2, 62]]
        self.allcommodity_ksps = [[[2, 1, 3], [2, 4, 3]], [[1, 2], [1, 3, 4, 2]]]
        self.allcommodity_notfstksps = [[[2, 4, 3]], [[1, 3, 4, 2]]]
        self.combination = [[[2, 1, 3], [1, 2]], [[2, 1, 3], [1, 3, 4, 2]], [[2, 4, 3], [1, 2]], [[2, 4, 3], [1, 3, 4, 2]]]
        self.grouping = self.combination[0]

        return self.grouping
   
    def gridgraph(self, n, m): # gridgraph生成
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
        self = G
        for i, j in G.edges(): # capaciy設定
            random.seed()
            v = random.randrange(300, 1000, 100)
            G.adj[i][j]['capacity'] = v
        # nx.write_gml(G,"/Users/takahashihimeno/Documents/result/graph.gml")
        for i, j in G.edges():
            G.add_edge(i, j, flow = dict(self.flow_initialValues), x = dict(), x_kakai = dict())
        pos = nx.spring_layout(G)
        capacity = nx.get_edge_attributes(G, 'capacity')
        edge_list = list(enumerate(G.edges()))
        edges_notindex = []
        for i in range(len(edge_list)):
            edges_notindex.append(edge_list[i][1])
        return G, pos, capacity, edge_list, edges_notindex

    def debaggraph(self,n,m): # デバッグ用
        G = nx.grid_2d_graph(n, m)
        mapping = dict(zip(G, range(1, 100)))
        G = nx.relabel_nodes(G, mapping)
        G = nx.DiGraph(G) # 有向グラフに変換
        #define capacity
        G.adj[1][3]['capacity'] = 600
        G.adj[1][2]['capacity'] = 700
        G.adj[2][1]['capacity'] = 700
        G.adj[2][4]['capacity'] = 400
        G.adj[3][1]['capacity'] = 700
        G.adj[3][4]['capacity'] = 800
        G.adj[4][2]['capacity'] = 800
        G.adj[4][3]['capacity'] = 900
        return G
 
    def generate_commodity(self, commodity): # 品種の定義(numpy使用)
        determin_st = []
        commodity_dictlist = []
        commodity_list = []
        for i in range(commodity): # commodity generate
            commodity_dict = {}
            s , t = tuple(random.sample(self.G.nodes, 2)) # source，sink定義
            demand = random.randint(10, 100) # demand設定
            tentative_st = [s,t] 
            while True:
                if tentative_st in determin_st:
                    s , t = tuple(random.sample(self.G.nodes, 2)) # source，sink再定義
                    tentative_st = [s,t]
                else:
                    break 
            determin_st.append(tentative_st) # commodity決定
            commodity_dict["id"] = i
            commodity_dict["source"] = s 
            commodity_dict["sink"] = t
            commodity_dict["demand"] = demand
            # print("commodity_list",commodity_list)
            # commodity_list.append(commodity_dict)

            commodity_list.append([s,t,demand])
            commodity_dictlist.append(commodity_dict)
            # print("commodity_dict",commodity_dict)
            # print("commodity_list",commodity_list)
        commodity_list.sort(key=lambda x: -x[2]) # demand大きいものから降順
        
        # with open('/Users/takahashihimeno/Documents/result/variety.csv','w') as f:
        #     writer=csv.writer(f,lineterminator='\n')
        #     writer.writerows(variety)

        return commodity_dictlist,commodity_list

    def search_ksps(self, K, G, commodity,commodity_list): # 各品種のkspsを求める
        allcommodity_ksps = []
        allcommodity_notfstksps = []
        for i in range(commodity):
            X = nx.shortest_simple_paths(G, commodity_list[i][0], commodity_list[i][1]) # Yen's algorithm
            ksps_list = []
            for counter, path in enumerate(X):            
                ksps_list.append(path)
                if counter == K - 1: 
                    break
            allcommodity_ksps.append(ksps_list)
            subksps_list = copy.deepcopy(ksps_list)
            subksps_list.pop(0)
            allcommodity_notfstksps.append(subksps_list)
        return allcommodity_ksps,allcommodity_notfstksps

    def searh_combination(self, allcommodity_ksps): # 組み合わせを求める
        combination = []
        q = [*product(*allcommodity_ksps)] # 全通りの組み合わせ
        for conbi in q:
            conbi = list(conbi)
            # print(conbi)
            combination.append(conbi)
        return combination   
    #####

    def reset(self): # 変数の初期化。最短経路の組み合わせに初期化する。doneがTrueになったときに呼び出される。
        # self.time = 0
        # self.grouping = combination[0]
        # return self.get_observation()
        self.time = 0
        self.grouping = self.get_random_grouping() # 本番用
        # self.grouping = self.get_random_grouping_debug() # デバッグ用
        # print("")
        # print("self.capacity_list",self.capacity_list)
        # print("self.edge_list",self.edge_list)
        # print("edges_notindex",self.edges_notindex)
        # print("graph",self.G.number_of_nodes())
        # print("self.commodity_list",self.commodity_list)
        # print("self.allcommodity_ksps",self.allcommodity_ksps)
        # print("self.allcommodity_notfstksps",self.allcommodity_notfstksps)
        # print("self.combination",self.combination)
        # print("self.candidate_list_in_reset",self.candidate_list)
        # print("")
        self.pair_list, self.nothingfstpath_pair_list = self.get_pair_list()
        reward = self.get_reward(self.grouping)

        return self.get_observation(),reward,self.G.number_of_nodes(),self.commodity_list
 
    def step(self, action): # 各ステップで実行される操作
        #actionの値にしたがってメンバーを入れ替える
        #⇒メソッドget_observationで入れ替え後の状態に対して観測データを得る
        #⇒メソッドget_rewardで報酬を計算する
        #⇒終了条件を満たしているかを判定する

        self.time += 1
        # step1: actionに戻づいてグループ分けを更新する
        print("self.grouping",self.grouping)
        print("action",action)
        print("self.candidate_list_in_step",self.candidate_list)
        # print("")
        self.grouping = self.exchange_path_action(self.grouping, action).copy()
        # print("grouping_after_axchangepath_action",self.grouping)
        # print("")
        
        # step2: 観測データ(observation)の計算
        observation = self.get_observation() # 入れ替え後の状態についての観測データを得る
        # print("")
        # print("observation",observation)
        # print("")

        # step3: 報酬(reward)の計算
        reward = self.get_reward(self.grouping) # 今のobservationの中の報酬が最大のものではなく、candidateからationの値で選ばれたもののcostが出てくる
        # print("")
        # print("reward",reward)
        # print("")
        
        # step4: 終了時刻を満たしているかの判定
        done = self.check_is_done()
        info = {}
        return observation, reward, done, info

    def get_observation(self): # 観測データを計算する関数　経路の組み替えによる負荷率をみたいから、pairでのobservationを確認する
        # grouping = self.grouping.copy()
        # candidate_list = []
        # for c in range(self.commodity):
        #     for j in range(len(allcommodity_notfstksps[c])):
        #         path = allcommodity_notfstksps[c][j] #変更する経路　ここをランダムにするかどうか？
        #         cost = self.get_reward(self.exchange_path_pair(grouping, c, path))
        #         candidate_list.append([c, path, cost])

        # self.candidate_list = sorted(candidate_list, key=lambda x:-x[2])[0:self.n_action] # costの大きい順に並び替えて、self.n_action個取り出す
        # return [cand[2] for cand in self.candidate_list] # cost(最大負荷率)をリスト化して返す　観測データを返したいから
        grouping = self.grouping.copy()
        candidate_list = []
        # for c in range(self.commodity): # notfstpathを使用した組み替え
        #     for j in range(len(self.allcommodity_notfstksps[c])):
        #         path = self.allcommodity_notfstksps[c][j] #変更する経路　ここをランダムにするかどうか？
        #         cost = self.get_reward(self.exchange_path_pair(grouping, c, path))
        #         candidate_list.append([c, path, cost])
        #         print("candidate_list",candidate_list)

        # if self.time == 0:
        #     for pair in self.nothingfstpath_pair_list:
        #         c, path = pair
        #         cost = self.get_reward(self.exchange_path_pair(grouping, c, path))
        #         candidate_list.append([c, path, cost])
        # else:
        #     for pair in self.pair_list:
        #         c, path = pair
        #         cost = self.get_reward(self.exchange_path_pair(grouping, c, path))
        #         candidate_list.append([c, path, cost])

        for pair in self.pair_list:
            c, path = pair
            cost = self.get_reward(self.exchange_path_pair(grouping, c, path))
            candidate_list.append([c, path, cost])

        self.candidate_list = sorted(candidate_list, key=lambda x:-x[2])[0:self.n_action] # costの大きい順に並び替えて、self.n_action個取り出す 最大負荷率が小さい順
        # print("self.candidate_list",self.candidate_list)
        return [cand[2] for cand in self.candidate_list] # cost(最大負荷率)をリスト化して返す　観測データを返したいから

    def exchange_path_action(self, grouping, action): # actionの値に応じて経路を交換する
        new_grouping = grouping.copy()
        c, path, cost = self.candidate_list[action]
        new_grouping[c] = path
        return new_grouping

    def exchange_path_pair(self, grouping, c, path): # 経路を交換する
        new_grouping = grouping.copy() # grouping:[[1,2,3],[2,4]]
        # new_grouping = copy.deepcopy(grouping)
        new_grouping[c] = path
        return new_grouping

    def get_reward(self, grouping): # 報酬を計算する関数
        return -1 * self.MaxLoadFactor(grouping) # 報酬 = 最大負荷率*(-1)
    
    def MaxLoadFactor(self, grouping): # 最大負荷率を計算する関数
        # return sum([self.LoadFactor(grouping, group_id) for group_id in self.group_id_list])
        return max(self.LoadFactor(grouping))
    
    def LoadFactor(self, grouping): # 負荷率の計算
        # load = []
        # zo_combination = self.newzero_one(grouping)
        # for e in range(len(edge_list)): #容量制限
        #     load.append(sum((zo_combination[l][e])*(commodity_list[l][2])for l in range(commodity)) / capacity_list[edge_list[e][1]])
        # return load
        load = []
        zo_combination = self.newzero_one(grouping)
        for e in range(len(self.edge_list)): #容量制限
            load.append(sum((zo_combination[l][e])*(self.commodity_list[l][2])for l in range(self.commodity)) / self.capacity_list[self.edge_list[e][1]])
        # print("load",load)
        return load
    
    def newzero_one(self, grouping): # 経路をもとに辺の01変換処理
        # zo_combination = []
        # # flow_var_kakai = []
        # for l in range(commodity):
        #     x_kakai = len(edges_notindex)*[0]
        #     for a in range(len(grouping[l])):
        #         if a == len(grouping[l])-1:
        #             break
        #         set = (grouping[l][a],grouping[l][a+1])
        #         # print(set)
        #         idx = edges_notindex.index(set)
        #         x_kakai[idx] = 1
        #     # flow_var_kakai.append(x_kakai)
        #     zo_combination.append(x_kakai)
        # # zo_combination.append(flow_var_kakai)
        # return zo_combination
        zo_combination = []
        # flow_var_kakai = []
        for l in range(self.commodity):
            x_kakai = len(self.edges_notindex)*[0]
            for a in range(len(grouping[l])):
                if a == len(grouping[l])-1:
                    break
                set = (grouping[l][a],grouping[l][a+1])
                # print(set)
                idx = self.edges_notindex.index(set)
                x_kakai[idx] = 1
            # flow_var_kakai.append(x_kakai)
            zo_combination.append(x_kakai)
        # zo_combination.append(flow_var_kakai)
        return zo_combination
#####
# --------------------------------------------------------------------------------------- 

def test_environment():
    # 環境を初期化
    # env = min_maxload_KSPs_Env(G, capacity_list, edge_list, edges_notindex, commodity, commodity_list, allcommodity_ksps, allcommodity_notfstksps, combination)
    env = min_maxload_KSPs_Env(K, n_action, obs_low, obs_high, max_step, grid_node, range_commodity)

    # 環境をリセットして初期状態を取得
    initial_observation, initial_reward, number_of_nodes, commodity_list = env.reset()
    # print("initial_observation",initial_observation)
    # print("initial_reward",initial_reward)
    
    # 初期状態がNoneでないことを確認
    # assert initial_observation is not None
    # print("initial_observation",initial_observation)
    
    print("")
    print("----------------------------------------------------------------------------------------")
    print("")

    pair_list = env.get_pair_list()
    all_reward = []
    for i in range(100):
        action = random.randint(0, len(pair_list))
        observation, reward, done, info = env.step(action)
        print(observation, reward, done, info)
        all_reward.append(reward)
        print("")
        print("")
        if done:
            break  # エピソード終了条件を満たしたらループを終了
    
    print("graph",number_of_nodes)
    print("commodity_list",commodity_list)
    print("initial_reward",initial_reward)
    print("min(all_reward)",max(all_reward))
    print(all_reward)
    # エージェントが選択するアクションを指定
    # action = 0
    
    # 1ステップ進めて新しい状態、報酬、エピソード終了情報を取得
    # observation, reward, done, info = env.step(action)
    # print(observation, reward, done, info)
    # observation, reward, done, info = env.step(action)
    # print(observation, reward, done, info)
    
    # 新しい観測がNoneでないことを確認
    # assert observation is not None
    
    # 報酬が数値であることを確認
    # assert isinstance(reward, (int, float))
    
    # エピソードが終了していないことを確認
    # assert not done
    
    # infoが辞書であることを確認
    # assert isinstance(info, dict)
    
    # その他のアサーションを追加することも可能
    
    # 環境をクローズ
    env.close()

# --------------------------------------------------------------------------------------- 
K = 3 # パスの個数
n_action = 10 # candidateの個数
obs_low = -10 # 観測変数のスペース　下限
obs_high = 10 # 観測変数のスペース　上限
max_step = 5 # 
grid_node = 3 # gridgraphの列数範囲
range_commodity = 3 # 品種の範囲
# env = min_maxload_KSPs_Env(K, n_action, obs_low, obs_high, max_step, grid_node, range_commodity)
# テストを実行
test_environment()

# def gridgraph(n, m): # gridgraph生成
#     G = nx.grid_2d_graph(n, m) 
#     n = n*m
#     mapping = dict(zip(G, range(1, 100)))
#     G = nx.relabel_nodes(G, mapping)
#     G = nx.DiGraph(G) # 有向グラフに変換
#     G.number_of_area = n
#     G.area_nodes_dict = dict()
#     G.flow_initialValues = dict()
#     G.all_flows = list()
#     for a in range(n):
#         G.area_nodes_dict[a] = []
#     self=G
#     for i, j in G.edges(): # capaciy設定
#         random.seed()
#         v = random.randrange(300, 1000, 100)
#         G.adj[i][j]['capacity'] = v
#     # nx.write_gml(G,"/Users/takahashihimeno/Documents/result/graph.gml")
#     for i, j in G.edges():
#         G.add_edge(i, j, flow = dict(self.flow_initialValues), x = dict(), x_kakai = dict())
#     pos = nx.spring_layout(G)
#     capacity = nx.get_edge_attributes(G, 'capacity')
#     edge_list = list(enumerate(G.edges()))
#     edges_notindex = []
#     for i in range(len(edge_list)):
#         edges_notindex.append(edge_list[i][1])
#     return G, pos, capacity, edge_list, edges_notindex
# def randomgraph(n, p): # randomgraph生成
#     G = nx.fast_gnp_random_graph(n, p)
#     while True:
#         if nx.is_connected(G) == False: #強連結グラフ判定
#             G = nx.fast_gnp_random_graph(n, p)
#         else:
#             break
#         # print("次数", G.degree())
#     for i in range(n): # 次数3以上のグラフ作成
#         while True:
#             if nx.degree(G)[i] < 3:
#                 j = random.choice(range(n))
#                 while True:
#                     if i != j:
#                         G.add_edge(i, j)
#                         break
#                     else:
#                         j = random.choice(range(n))
#             else:
#                 break
#     G = nx.DiGraph(G) #有向グラフに変換
#     G.number_of_area = n
#     G.area_nodes_dict = dict()
#     G.flow_initialValues = dict()
#     G.all_flows = list()
#     for a in range(n):
#         G.area_nodes_dict[a] = []
#     self = G
#     for i, j in G.edges(): #capaciy設定
#         random.seed()
#         v = random.randrange(300, 1000, 100)
#         G.adj[i][j]['capacity'] = v
#     # nx.write_gml(G, "/Users/takahashihimeno/Documents/result/graph.gml")
#     for i, j in G.edges():
#         G.add_edge(i, j, flow = dict(self.flow_initialValues), x = dict(), x_kakai = dict())
#     pos = nx.circular_layout(G)
#     capacity = nx.get_edge_attributes(G, 'capacity')
#     edge_list = list(enumerate(G.edges()))
#     edges_notindex = []
#     for i in range(len(edge_list)):
#         edges_notindex.append(edge_list[i][1])
#     return G, pos, capacity, edge_list, edges_notindex   
# def show(G, pos): # グラフの参照
#     nx.draw_networkx(G, pos, with_labels = True, alpha = 0.5)
#     plt.show()
# def generate_commodity(commodity): # 品種の定義(numpy使用)
#     determin_st = []
#     commodity_dictlist = []
#     commodity_list = []
#     for i in range(commodity): # commodity generate
#         commodity_dict = {}
#         s , t = tuple(random.sample(G.nodes, 2)) # source，sink定義
#         demand = random.randint(10, 100) # demand設定
#         tentative_st = [s,t] 
#         while True:
#             if tentative_st in determin_st:
#                 s , t = tuple(random.sample(G.nodes, 2)) # source，sink再定義
#                 tentative_st = [s,t]
#             else:
#                 break 
#         determin_st.append(tentative_st) # commodity決定
#         commodity_dict["id"] = i
#         commodity_dict["source"] = s 
#         commodity_dict["sink"] = t
#         commodity_dict["demand"] = demand
#         # print("commodity_list",commodity_list)
#         # commodity_list.append(commodity_dict)

#         commodity_list.append([s,t,demand])
#         commodity_dictlist.append(commodity_dict)
#         # print("commodity_dict",commodity_dict)
#         # print("commodity_list",commodity_list)
#     commodity_list.sort(key=lambda x: -x[2]) # demand大きいものから降順
    
#     # with open('/Users/takahashihimeno/Documents/result/variety.csv','w') as f:
#     #     writer=csv.writer(f,lineterminator='\n')
#     #     writer.writerows(variety)

#     return commodity_dictlist,commodity_list
# def generate_numpy_commodity(commodity): # 品種の定義(numpy使用)
#     determin_st = []
#     commodity_dictlist = np.empty(0)
#     commodity_list = np.empty(0)
#     # commodity_list = []
#     for i in range(commodity): # commodity generate
#         commodity_dict = {}
#         s , t = tuple(random.sample(G.nodes, 2)) # source，sink定義
#         demand = random.randint(10, 100) # demand設定
#         tentative_st = [s,t] 
#         while True:
#             if tentative_st in determin_st:
#                 s , t = tuple(random.sample(G.nodes, 2)) # source，sink再定義
#                 tentative_st = [s,t]
#             else:
#                 break 
#         determin_st.append(tentative_st) # commodity決定
#         commodity_dict["id"] = i
#         commodity_dict["source"] = s 
#         commodity_dict["sink"] = t
#         commodity_dict["demand"] = demand
#         # print("commodity_list",commodity_list)
#         # commodity_list.append(commodity_dict)
#         commodity_list = np.append(commodity_list, s)
#         commodity_list = np.append(commodity_list, t)
#         commodity_list = np.append(commodity_list, demand)
#         commodity_dictlist = np.append(commodity_dictlist, commodity_dict)
#         # print("commodity_dict",commodity_dict)
#         # print("commodity_list",commodity_list)
#     commodity_list = commodity_list.reshape(commodity, 3)
#     commodity_list = commodity_list[np.argsort(commodity_list[:, 2])[::-1]] # demand大きいものから降順
#     # with open('/Users/takahashihimeno/Documents/result/variety.csv','w') as f:
#     #     writer=csv.writer(f,lineterminator='\n')
#     #     writer.writerows(variety)

#     return commodity_dictlist,commodity_list
# def search_ksps(K,G,commodity,commodity_list): # 各品種のkspsを求める
#     allcommodity_ksps = []
#     allcommodity_notfstksps = []
#     for i in range(commodity):
#         X = nx.shortest_simple_paths(G, commodity_list[i][0], commodity_list[i][1]) # Yen's algorithm
#         ksps_list = []
#         for counter, path in enumerate(X):            
#             ksps_list.append(path)
#             if counter == K - 1: 
#                 break
#         allcommodity_ksps.append(ksps_list)
#         subksps_list = copy.deepcopy(ksps_list)
#         subksps_list.pop(0)
#         allcommodity_notfstksps.append(subksps_list)
#     return allcommodity_ksps,allcommodity_notfstksps
# def searh_combination(allcommodity_ksps): # 組み合わせを求める
#     combination = []
#     q = [*product(*allcommodity_ksps)] # 全通りの組み合わせ
#     for conbi in q:
#         conbi = list(conbi)
#         # print(conbi)
#         combination.append(conbi)
#     return combination
# def zero_one(edges_notindex, combination, commodity_list): # 経路をもとに辺の01変換処理
#     zo_combination = []
#     for i in range(len(combination)):
#         flow_var_kakai = []
#         # zo_combination = []
#         x_kakai = len(edges_notindex)*[0]
#         for l in range(len(commodity_list)):
#             for a in range(len(combination[i][l])):
#                 # print(combination[i][l['id']])
#                 if a == len(combination[i][l])-1:
#                     break
#                 set = (combination[i][l][a],combination[i][l][a+1])
#                 # print(set)
#                 idx = edges_notindex.index(set)
#                 x_kakai[idx] = 1
#             # zo_combination.append(x_kakai)
#             flow_var_kakai.append(x_kakai)
#         zo_combination.append(flow_var_kakai)
#         # print(zo_combination)
#     return zo_combination

# print("--------------------------------------------------")
# print("")

# グラフの定義
# retsu = 3 # 行列
# node = 10 #　ノード数
# p = 0.15 #　リンクを張る密度
# # G, pos, capacity_list, edge_list, edges_notindex = gridgraph(retsu, retsu) # gridgraph
# G, pos, capacity_list, edge_list, edges_notindex = randomgraph(node, p) #　randomgraph
# # show(G,pos)
# print("capacity_list:",capacity_list)
# print("edge_list:",edge_list) # [(0, (1, 3)), (1, (1, 2)), (2, (2, 1)), (3, (2, 4)), (4, (3, 1)), (5, (3, 4)), (6, (4, 2)), (7, (4, 3))]
# print("edges_notindex:",edges_notindex) # [(1, 3), (1, 2), (2, 1), (2, 4), (3, 1), (3, 4), (4, 2), (4, 3)]
# # print(capacity_list[edge_list[0][1]])
# print("")

# 品種の定義
# commodity = 2 # 品種数
# # commodity_dictlist, commodity_list = generate_commodity(commodity)
# print("commodity_dictlist:",commodity_dictlist)
# print("commodity_list:",commodity_list)
# # print(commodity_list[0]['sink'])
# print("")

# KSPsの探索
# K = 3 # パスの個数
# allcommodity_ksps, allcommodity_notfstksps = search_ksps(K,G,commodity,commodity_list)
# print("allcommodity_ksps:",allcommodity_ksps)
# print("allcommodity_notfstksps:",allcommodity_notfstksps)
# print("")

# 組み合わせの探索
# combination = searh_combination(allcommodity_ksps)
# print("combination:",combination)
# # print("The number of combination:",len(combination))
# zo_combination = zero_one(edges_notindex, combination, commodity_list)
# print("zero_one combination",zo_combination)

# print("")
# print("--------------------------------------------------")

# allcommodity_ksps = np.array(allcommodity_ksps)
# combination = np.array(combination)

# env = min_maxload_KSPs_Env(G, capacity_list, edge_list, edges_notindex, commodity, commodity_list, allcommodity_ksps, allcommodity_notfstksps, combination, zo_combination)
# env = min_maxload_KSPs_Env(G, capacity_list, edge_list, edges_notindex, commodity, commodity_list, allcommodity_ksps, allcommodity_notfstksps, combination)
