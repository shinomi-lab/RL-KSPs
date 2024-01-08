#"shufflelunch.py"から学習構築構築したmainのプログラムファイル

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
import random
from Exact_Solution import Solve_exact_solution
from Simple_ksp import Solve_simple_ksp
from flow import Flow
import graph_making
import os
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Input
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import rl.callbacks
import datetime

class min_maxload_KSPs_Env(gym.core.Env): # クラスの定義
    #gymで強化学習環境を作る場合はstep,reset,render,close,seedメソッドを実装

    def __init__(self, K, n_action, obs_low, obs_high, max_step, node_l, node_h, range_commodity_l, range_commodity_h, sample_size,capa_l,capa_h,demand_l,demand_h,graph_model, degree):
        self.K = K # kspsの本数
        self.node = random.randint(node_l, node_h)
        self.commodity = random.randint(range_commodity_l, range_commodity_h)
        self.sample_size = sample_size
        self.capa_l = capa_l
        self.capa_h = capa_h
        self.demand_l = demand_l
        self.demand_h = demand_h
        self.graph_model = graph_model
        self.degree = degree

        # graphの定義
        if(graph_model == 'grid'):
            self.G = graph_making.Graphs(self.commodity) #品種数,areaの品種数　この時点では何もグラフが作られていない　インスタンスの作成？
            self.G.gridMaker(self.G,self.node*self.node,self.node,self.node,0.1,self.capa_l,self.capa_h) #G,エリア数,エリアのノード数,列数,行数,ε浮動小数点
        if(graph_model == 'random'):
            self.G = graph_making.Graphs(self.commodity) #品種数,areaの品種数
            self.G.randomGraph(self.G, self.degree, self.node, self.capa_l, self.capa_h) # 5 is each node is joined with its k nearest neighbors in a ring topology. 5はノード数に関係しそう　次数だから
        self.capacity_list = nx.get_edge_attributes(self.G,'capacity') # 全辺のcapacityの値を辞書で取得
        self.edge_list = list(enumerate(self.G.edges()))
        self.edges_notindex = []
        for z in range(len(self.edge_list)):
            self.edges_notindex.append(self.edge_list[z][1])

        self.commodity_list = self.generate_commodity(self.commodity) # 品種作成
        self.random_ksps, self.allcommodity_ksps, self.allcommodity_notfstksps = self.search_ksps(self.K, self.G, self.commodity, self.commodity_list) # kspsの探索
        # self.combination = self.searh_combination(self.random_ksps) # 抽出バージョン
        self.combination = self.searh_combination(self.allcommodity_ksps)
        self.grouping = self.combination[0] #初期パターン（最短経路の組み合わせ）
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
        # print("self.time_in_check_is_done",self.time)
        return self.time == self.max_step # 最大数に達したら終了する

    def get_pair_list(self): # 品種と経路のペアを列挙
        pair_list = []
        for c in range(self.commodity):
            for p in range(len(self.allcommodity_ksps[c])):
                path = self.allcommodity_ksps[c][p]
                pair_list.append([c, path])
        # print("pair_list",pair_list)
        return pair_list

   ##### resetで必要な関数 #####
    def get_random_grouping(self):
        self.node = random.randint(node_l, node_h) # ランダムでgridgraphの行列数を決定
        self.commodity = random.randint(range_commodity_l, range_commodity_h) # ランダムで品種数を決定

        if(graph_model == 'grid'):
            self.G = graph_making.Graphs(self.commodity) #品種数,areaの品種数　この時点では何もグラフが作られていない　インスタンスの作成？
            self.G.gridMaker(self.G,self.node*self.node,self.node,self.node,0.1,self.capa_l,self.capa_h) #G,エリア数,エリアのノード数,列数,行数,ε浮動小数点
        if(graph_model == 'random'):
            self.G = graph_making.Graphs(self.commodity) #品種数,areaの品種数
            self.G.randomGraph(self.G, self.degree , self.node, self.capa_l, self.capa_h) # 5 is each node is joined with its k nearest neighbors in a ring topology. 5はノード数に関係しそう　次数だから
        self.capacity_list = nx.get_edge_attributes(self.G,'capacity') # 全辺のcapacityの値を辞書で取得
        self.edge_list = list(enumerate(self.G.edges()))
        self.edges_notindex = []
        for z in range(len(self.edge_list)):
            self.edges_notindex.append(self.edge_list[z][1])

        self.commodity_list = self.generate_commodity(self.commodity) # 品種作成
        self.random_ksps, self.allcommodity_ksps, self.allcommodity_notfstksps = self.search_ksps(self.K, self.G, self.commodity, self.commodity_list) # kspsの探索
        # self.combination = self.searh_combination(self.random_ksps) # 抽出バージョン
        self.combination = self.searh_combination(self.allcommodity_ksps) # 組み合わせを求める
        self.grouping = self.combination[0] # 初期stateは最短経路の組み合わせ

        return self.grouping
     
    def generate_commodity(self, commodity): # 品種の定義(numpy使用)
        determin_st = []
        commodity_list = []
        for i in range(commodity): # commodity generate
            commodity_dict = {}
            s , t = tuple(random.sample(self.G.nodes, 2)) # source，sink定義
            demand = random.randint(self.demand_l, self.demand_h) # demand設定
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

            commodity_list.append([s,t,demand])
        commodity_list.sort(key=lambda x: -x[2]) # demand大きいものから降順

        return commodity_list

    def search_ksps(self, K, G, commodity,commodity_list): # 各品種のkspsを求める
        allcommodity_ksps = []
        random_ksps = []
        allcommodity_notfstksps = []
        for i in range(commodity):
            X = nx.shortest_simple_paths(G, commodity_list[i][0], commodity_list[i][1]) # Yen's algorithm
            ksps_list = []
            for counter, path in enumerate(X):            
                ksps_list.append(path)
                if counter == K - 1: 
                    break
            # 経路のサンプリング
            random_sample = random.sample(ksps_list, self.sample_size)
            random_ksps.append(random_sample)
            allcommodity_ksps.append(ksps_list)

            subksps_list = copy.deepcopy(ksps_list)
            subksps_list.pop(0)
            allcommodity_notfstksps.append(subksps_list)
        return random_ksps,allcommodity_ksps,allcommodity_notfstksps
  
    def searh_combination(self, allcommodity_ksps): # 最短の組み合わせを求める
        comb = []
        L = len(allcommodity_ksps)
        for z in range(L):
            # print(allcommodity_ksps[i][0])
            comb.append(allcommodity_ksps[z][0])
        combination = [comb]
        # print(combination)
        return combination   

    def reset(self): # 変数の初期化。エピソードの初期化。doneがTrueになったときに呼び出される。
        self.time = 0 # ステップ数の初期化
        self.grouping = self.get_random_grouping()
        self.pair_list = self.get_pair_list() # アクションの範囲を求める
        # initial_reward = self.get_reward(self.grouping) # 初期状態の報酬（最短経路の組み合わせの最小の最大負荷率）

        # return self.get_observation(), initial_reward, self.G.number_of_nodes(), self.commodity_list
        return self.get_observation()
 
    def step(self, action): # 各ステップで実行される操作
        #actionの値にしたがってメンバーを入れ替える
        #⇒メソッドget_observationで入れ替え後の状態に対して観測データを得る
        #⇒メソッドget_rewardで報酬を計算する
        #⇒終了条件を満たしているかを判定する

        self.time += 1
        # step1: actionに戻づいてグループ分けを更新する
        self.grouping = self.exchange_path_action(self.grouping, action).copy()
        
        # step2: 観測データ(observation)の計算
        observation = self.get_observation() # 入れ替え後の状態についての観測データを得る

        # step3: 報酬(reward)の計算
        reward = self.get_reward(self.grouping) # 今のobservationの中の報酬が最大のものではなく、candidateからationの値で選ばれたもののcostが出てくる
        
        # step4: 終了時刻を満たしているかの判定
        done = self.check_is_done()
        info = {}
        return observation, reward, done, info

    def get_observation(self): # 観測データを計算する関数　経路の組み替えによる負荷率をみたいから、pairでのobservationを確認する
        grouping = self.grouping.copy()
        candidate_list = []

        for pair in self.pair_list:
            c, path = pair # 品種と経路のペアリストから一つずつ取り出す
            cost = self.get_reward(self.exchange_path_pair(grouping, c, path)) # 今のstateから対象の品種cに対して経路をpathに変更して報酬を計算する
            candidate_list.append([c, path, cost])

        self.candidate_list = sorted(candidate_list, key=lambda x:-x[2])[0:self.n_action] # costの大きい順に並び替えて、self.n_action個取り出す 最大負荷率が小さい順
        # print("self.candidate_list",self.candidate_list)
        return [cand[2] for cand in self.candidate_list] # cost(最大負荷率)をリスト化して返す　観測データを返したいから

    def exchange_path_action(self, grouping, action): # actionの値に応じて経路を交換する
        new_grouping = grouping.copy()
        c, path, cost = self.candidate_list[action]
        new_grouping[c] = path
        return new_grouping

    def exchange_path_pair(self, grouping, c, path): # candidate_listを作成するために経路を交換させる
        new_grouping = grouping.copy() # grouping:[[1,2,3],[2,4]]
        new_grouping[c] = path
        return new_grouping

    def get_reward(self, grouping): # 報酬を計算する関数
        return -1 * self.MaxLoadFactor(grouping) # 報酬 = 最大負荷率*(-1)
    
    def MaxLoadFactor(self, grouping): # 最大負荷率を計算する関数
        # return sum([self.LoadFactor(grouping, group_id) for group_id in self.group_id_list])
        return max(self.LoadFactor(grouping))
    
    def LoadFactor(self, grouping): # 負荷率の計算
        loads = []
        zo_combination = self.zero_one(grouping)
        for e in range(len(self.edge_list)): #容量制限
            load = sum((zo_combination[l][e])*(self.commodity_list[l][2])for l in range(self.commodity)) / self.capacity_list[self.edge_list[e][1]]
            if load >= 1:
                load = -1000
            loads.append(load)
            # load.append(sum((zo_combination[l][e])*(self.commodity_list[l][2])for l in range(self.commodity)) / self.capacity_list[self.edge_list[e][1]])
        # print("load",load)
        return loads
    
    def zero_one(self, grouping): # 経路をもとに辺の01変換処理
        zo_combination = []
        for l in range(self.commodity):
            x_kakai = len(self.edges_notindex)*[0]
            for a in range(len(grouping[l])):
                if a == len(grouping[l])-1:
                    break
                set = (grouping[l][a],grouping[l][a+1])
                # print(set)
                idx = self.edges_notindex.index(set)
                x_kakai[idx] = 1
            zo_combination.append(x_kakai)
        return zo_combination

# カスタムコールバックを作成
class CustomEpisodeLogger(rl.callbacks.Callback):
    def __init__(self):
        self.episode = 0
        self.start_time = 0 # エピソードごとの処理時間
        self.rewards = {}  # エピソードごとの報酬を保存する辞書
        self.objective_values = []
        self.objective_time = []
        self.objective_values_ksp = []
        self.objective_time_ksp = []
        self.apploximatesolutions = []
        self.apploximatetime = []

    def on_episode_begin(self, episode, logs):
        self.episode = episode
        self.start_time = time.time() # エピソード開始時間の取得
        # 新しいエピソードが始まるたびに新しいリストを作成
        self.rewards[self.episode] = []

    def on_step_end(self, step, logs):
        reward = logs['reward']
        self.rewards[self.episode].append(reward)

    def on_episode_end(self, episode, logs):
        # エピソードが終了した際に呼ばれる
        # エピソード終了時に1回だけログを表示
        end_time = time.time() # エピソード終了時間の取得
        elapsed_time = end_time - self.start_time # 処理時間計算
        apploximate_solution = self.rewards[self.episode][-1]*(-1)
        self.apploximatesolutions.append(apploximate_solution)
        self.apploximatetime.append(elapsed_time)
        steps = logs['nb_steps']
        print(f"Episode {self.episode}: approximate_solution: {apploximate_solution}, steps: {steps}, time: {elapsed_time:.2f} seconds") # 独自のログ出力

        # ファイルにデータを書き込む
        with open('commodity_data.csv','w') as f:
            writer=csv.writer(f,lineterminator='\n')
            writer.writerows(self.env.commodity_list) # 品種の保存
        nx.write_gml(self.env.G, "graph.gml") # グラフの保存

        # approximateの書き込み
        with open(approx_file_name, 'a', newline='') as f:
            out = csv.writer(f)
            out.writerow([self.episode, apploximate_solution, elapsed_time])

        # 厳密解を求める
        E = Solve_exact_solution(self.episode,solver_type,exact_file_name) # Exact_Solution.pyの厳密解クラスの呼び出し
        objective_value,objective_time = E.solve_exact_solution_to_env() # 厳密解を計算
        self.objective_values.append(objective_value) # 厳密解情報を格納
        self.objective_time.append(objective_time)

        # 単純kspを求める
        E = Solve_simple_ksp(self.episode,solver_type,simpleksp_file_name,K,objective_value) # Simple_ksp.pyの厳密解クラスの呼び出し
        objective_value_ksp,objective_time_ksp = E.solve_simple_ksp_to_env() # 厳密解を計算
        self.objective_values_ksp.append(objective_value_ksp) # 厳密解情報を格納
        self.objective_time_ksp.append(objective_time_ksp)


def test_environment():
    # 環境を初期化
    env = min_maxload_KSPs_Env(K, n_action, obs_low, obs_high, max_step, node_l, node_h, range_commodity_l, range_commodity_h, sample_size,capa_l,capa_h,demand_l,demand_h,degree)
    print("set env")
    # 環境をリセットして初期状態を取得
    # initial_observation, initial_reward, number_of_nodes, commodity_list = env.reset()
    initial_observation = env.reset()
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
        if n_action<=len(pair_list):
            action = random.randint(0, n_action-1)
        else:
            action = random.randint(0, len(pair_list)-1)
        observation, reward, done, info = env.step(action)
        print(observation, reward, done, info)
        all_reward.append(reward)
        print("")
        print("")
        if done:
            break  # エピソード終了条件を満たしたらループを終了
    
    # print("graph",number_of_nodes)
    # print("commodity_list",commodity_list)
    # print("initial_reward",initial_reward)
    print("initial_observation",initial_observation)
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

graph_model = "random"
random.seed(1) #ランダムの固定化
solver_type = "SCIP"
result_model = "graph2" # graph1:厳密解との比較　graph2:単純kspとの比較

K = 10 # パスの個数
n_action = 20# candidateの個数
obs_low = -20 # 観測変数のスペース　下限
obs_high = 20 # 観測変数のスペース　上限
node_l = 20 # gridgraphの列数範囲
node_h = 100 # gridgraphの列数範囲
range_commodity_l = 5 # 品種の範囲
range_commodity_h = 5 # 品種の範囲
sample_size = 5  # 抽出する要素の数
capa_l = 1000 # capacityの範囲
capa_h = 10000 # capacityの範囲
demand_l = 1 # 需要量の範囲
demand_h = 500 # 需要量の範囲
degree = 3

ln_episodes =  10000 # 訓練エピソード数
max_step =  20 # 訓練時の最大step数
nb_episodes = 3 # テストエピソード数
nb_max_episode_steps = 20 # テスト時のstep数

# ln_episodes_list = [5000,10000,15000,20000,25000,30000,35000,40000,45000,50000]
ln_episodes_list = [5000]


for kurikaeshi in range(len(ln_episodes_list)):
    ln_episodes = ln_episodes_list[kurikaeshi]

    print("start set env")
    env = min_maxload_KSPs_Env(K, n_action, obs_low, obs_high, max_step, node_l, node_h, range_commodity_l, range_commodity_h, sample_size,capa_l,capa_h,demand_l,demand_h,graph_model,degree) # 実行
    print("finish set env")
    # test_environment() # テストを実行

    # ニューラルネットワークの構造を定義
    model = Sequential() # モデルの構築
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape)) # 入力層
    model.add(Dense(128)) # 中間層
    model.add(Activation('relu')) # 中間層の活性化関数定義
    model.add(Dense(n_action)) # 出力層
    model.add(Activation('linear')) # 出力層の活性化関数定義　恒等関数
    print("model.summary()",model.summary()) # モデルの定義をコンソールに出力

    # モデルのコンパイル
    memory = SequentialMemory(limit=50000, window_length=1) # メモリの用意
    policy = BoltzmannQPolicy(tau=1.) # ポリシーの設定
    dqn = DQNAgent(model=model, nb_actions=n_action, memory=memory, nb_steps_warmup=50, target_model_update=1e-2, policy=policy) # エージェントの作成
    dqn.compile(Adam(lr=1e-3), metrics=['mae']) # エージェントをコンパイル

    # 訓練
    history = dqn.fit(env, nb_steps=ln_episodes, visualize=True, verbose=2, nb_max_episode_steps=max_step)
    
    # 厳密解のファイルを用意
    exact_file_name = f'exactsolution_{kurikaeshi}_{range_commodity_l}.csv'
    with open(exact_file_name, 'w') as f:
        out = csv.writer(f)
    # 近似解のファイルを用意
    approx_file_name = f'approximatesolution_{kurikaeshi}_{range_commodity_l}.csv'
    with open(approx_file_name, 'w') as f:
        out = csv.writer(f)
    # 単純kspのファイルを用意
    simpleksp_file_name = f'simpleksp_{kurikaeshi}_{range_commodity_l}.csv'
    with open(simpleksp_file_name, 'w') as f:
        out = csv.writer(f)
    
    random.seed(7) #ランダムの固定化　テスト条件の固定
    episode_logger = CustomEpisodeLogger()
    # テスト時にカスタムコールバックを使用してエピソードごとの処理時間を取得
    dqn.test(env, nb_episodes=nb_episodes, nb_max_episode_steps=nb_max_episode_steps, visualize=False, callbacks=[episode_logger], verbose=0)
    env.close()

    if (result_model == 'graph1'):
        now = datetime.datetime.now()
        # グラフ保存先ディレクトリ
        save_dir = 'episode_graphs/'
        # ディレクトリが存在しない場合は作成
        os.makedirs(save_dir, exist_ok=True)

        # stepごとの平均reward推移
        mean_reward_list = []
        for i in range(len(episode_logger.rewards[0])):
            heikin = 0
            for j in range(len(episode_logger.rewards)):
                heikin = heikin + episode_logger.rewards[j][i]
            mean_reward_list.append((heikin / nb_episodes) * -1)
        x = list(range(1, nb_max_episode_steps + 1))
        plt.plot(x, mean_reward_list, label='N={}'.format(nb_episodes))
        plt.xlabel('step', fontsize=18)
        plt.ylabel('mean reward', fontsize=18)
        plt.legend(loc='upper right', fontsize=18)
        plt.xticks(x)

        # グラフをエピソードごとに保存
        filename = f"episode_{kurikaeshi + 1}_{range_commodity_l}_{now.strftime('%Y-%m-%d_%H-%M-%S')}_step_reward.png"

        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.clf()

        # 厳密解と近似解の比較
        y1 = episode_logger.objective_values # 厳密解
        y2 = episode_logger.apploximatesolutions # 近似解
        x = np.arange(len(y1)) # x軸の設定
        valid_indices1 = [i for i, value in enumerate(y1) if value is not None]
        valid_indices2 = [i for i, value in enumerate(y2) if value is not None]
        # 有効なデータのみを抽出
        valid_data1 = [y1[i] for i in valid_indices1]
        valid_data2 = [y2[i] for i in valid_indices2]
        valid_x1 = [x[i] for i in valid_indices1]
        valid_x2 = [x[i] for i in valid_indices2]
        # プロット
        plt.plot(valid_x1, valid_data1, marker='o', linestyle='-', label='exactsolution')
        plt.plot(valid_x2, valid_data2, marker='o', linestyle='-', label='approximatesolution')
        # ラベルや凡例の追加
        plt.xlabel('episode')
        plt.ylabel('value')
        # plt.title('二つのリストのプロット')
        plt.legend() # 凡例を表示

        # グラフをエピソードごとに保存
        filename = f"episode_{kurikaeshi + 1}_{range_commodity_l}_{now.strftime('%Y-%m-%d_%H-%M-%S')}_value.png"
        
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.clf()

        #近似率の算出
        apploximate_rate = []
        for i in range(nb_episodes):
            if y1[i] is None:
                apploximaterate = 110
            elif y2[i] is None:
                apploximaterate = 0
            else:
                apploximaterate = y1[i]/y2[i]*100
            apploximate_rate.append(apploximaterate)
        x = list(range(1, nb_episodes + 1))
        plt.plot(x, apploximate_rate, label='approximate rate')
        # ラベルや凡例の追加
        plt.xlabel('episode')
        plt.ylabel('value')
        # plt.title('二つのリストのプロット')
        plt.legend() # 凡例を表示

        # グラフをエピソードごとに保存
        filename = f"episode_{kurikaeshi + 1}_{range_commodity_l}_{now.strftime('%Y-%m-%d_%H-%M-%S')}_rate.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.clf()

        # 計算時間の比較
        y1 = episode_logger.objective_time # 厳密解の処理時間
        y2 = episode_logger.apploximatetime # 近似解の処理時間
        x = np.arange(len(y1)) # x軸の設定
        valid_indices1 = [i for i, value in enumerate(y1) if value is not None]
        valid_indices2 = [i for i, value in enumerate(y2) if value is not None]
        # 有効なデータのみを抽出
        valid_data1 = [y1[i] for i in valid_indices1]
        valid_data2 = [y2[i] for i in valid_indices2]
        valid_x1 = [x[i] for i in valid_indices1]
        valid_x2 = [x[i] for i in valid_indices2]
        # プロット
        plt.plot(valid_x1, valid_data1, marker='o', linestyle='-', label='exactsolution time')
        plt.plot(valid_x2, valid_data2, marker='o', linestyle='-', label='approximatesolution time')
        # ラベルや凡例の追加
        plt.xlabel('episode')
        plt.ylabel('s')
        # plt.title('二つのリストのプロット')
        plt.legend() # 凡例を表示

        # グラフをエピソードごとに保存
        filename = f"episode_{kurikaeshi + 1}_{range_commodity_l}_{now.strftime('%Y-%m-%d_%H-%M-%S')}_time.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.clf()

    if (result_model == 'graph2'):
        now = datetime.datetime.now()
        # グラフ保存先ディレクトリ
        save_dir = 'episode_graphs_simple/'
        # ディレクトリが存在しない場合は作成
        os.makedirs(save_dir, exist_ok=True)
        
        #近似率の算出
        approximate_rate = []
        approximate_rate_LR = []
        y1 = episode_logger.objective_values # 厳密解
        y2 = episode_logger.apploximatesolutions # LRksp
        y3 = episode_logger.objective_values_ksp # 単純ksp

        for i in range(nb_episodes):
            if y1[i] is None:
                approximaterate = approximaterate_LR = 110
            elif y2[i] is None:
                approximaterate_LR = 0
            elif y3[i] is None:
                approximaterate = 0
            else:
                approximaterate_LR = y1[i]/y2[i]*100
                approximaterate = y1[i]/y3[i]*100
            approximate_rate.append(approximaterate)
            approximate_rate_LR.append(approximaterate_LR)
        x = np.arange(len(y1)) # x軸の設定


        valid_indices1 = [i for i, value in enumerate(approximate_rate) if value is not None]
        valid_indices2 = [i for i, value in enumerate(approximate_rate_LR) if value is not None]
        # 有効なデータのみを抽出
        valid_data1 = [approximate_rate[i] for i in valid_indices1]
        valid_data2 = [approximate_rate_LR[i] for i in valid_indices2]
        valid_x1 = [x[i] for i in valid_indices1]
        valid_x2 = [x[i] for i in valid_indices2]
        # プロット
        plt.plot(valid_x1, valid_data1, marker='o', linestyle='-', label='Simple ksp')
        plt.plot(valid_x2, valid_data2, marker='o', linestyle='-', label='LP ksp')


        # plt.plot(x, approximate_rate, label='Simple ksp')
        # plt.plot(x, approximate_rate_LR, label='LR ksp')

        # ラベルや凡例の追加
        plt.xlabel('episode')
        plt.ylabel('approximate rate[%]')
        # plt.title('二つのリストのプロット')
        plt.legend() # 凡例を表示
        # グラフをエピソードごとに保存
        filename = f"episode_{kurikaeshi + 1}_{range_commodity_l}_{now.strftime('%Y-%m-%d_%H-%M-%S')}_compare_rate.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.clf()

        # 計算時間の比較
        y1 = episode_logger.objective_time_ksp # 単純kspの処理時間
        y2 = episode_logger.apploximatetime # 近似解の処理時間
        x = np.arange(len(y1)) # x軸の設定
        valid_indices1 = [i for i, value in enumerate(y1) if value is not None]
        valid_indices2 = [i for i, value in enumerate(y2) if value is not None]
        # 有効なデータのみを抽出
        valid_data1 = [y1[i] for i in valid_indices1]
        valid_data2 = [y2[i] for i in valid_indices2]
        valid_x1 = [x[i] for i in valid_indices1]
        valid_x2 = [x[i] for i in valid_indices2]
        # プロット
        plt.plot(valid_x1, valid_data1, marker='o', linestyle='-', label='Simple ksp')
        plt.plot(valid_x2, valid_data2, marker='o', linestyle='-', label='LP ksp')
        # ラベルや凡例の追加
        plt.xlabel('episode')
        plt.ylabel('time[s]')
        # plt.title('二つのリストのプロット')
        plt.legend() # 凡例を表示
        # グラフをエピソードごとに保存
        filename = f"episode_{kurikaeshi + 1}_{range_commodity_l}_{now.strftime('%Y-%m-%d_%H-%M-%S')}_compare_time.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.clf()