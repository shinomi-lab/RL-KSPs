#LP-ksps_envで使う厳密解を求めるためのプログラムファイル
from mip import *
import time
import csv
import networkx as nx
from flow import Flow

class Solve_exact_solution():
    def __init__(self, episode):
        self.episode = episode
        self.G = nx.read_gml("graph.gml",destringizer=int) # グラフの定義
        self.G.all_flows = list()

        with open('commodity_data.csv',newline='') as f: # 品種の読み込み
            self.commodity=csv.reader(f)
            self.commodity=[row for row in self.commodity]
            self.commodity=[[int(item) for item in row]for row in self.commodity]

        self.r_kakai = list(enumerate(self.G.edges()))
        self.commodity_count = 0
        self.tuples = []
        self.capacity = nx.get_edge_attributes(self.G, 'capacity')

        while(len(self.tuples)<len(self.commodity)): 
            s = self.commodity[self.commodity_count][0] # source
            t = self.commodity[self.commodity_count][1] # sink
            demand = self.commodity[self.commodity_count][2] # demand
            self.tuples.append((s,t))
            self.f = Flow(self.G,self.commodity_count,s,t,demand)
            self.G.all_flows.append(self.f)
            self.commodity_count +=1
        
    def solve_exact_solution_to_env(self):
        # 問題の定義(全体の下界)
        UELB_kakai = Model('UELB_kakai') # モデルの名前
        L_kakai = UELB_kakai.add_var('L_kakai',lb = 0, ub = 1)
        flow_var_kakai = []
        for l in range(len(self.G.all_flows)):
            x_kakai = [UELB_kakai.add_var('x{}_{}'.format(l,m), var_type = BINARY) for m, (i,j) in self.r_kakai]#enumerate関数のmとエッジ(i,j)
            flow_var_kakai.append(x_kakai) #品種エル(l)に対して全ての辺のIDを表す番号mがついている。中身は0-1
        UELB_kakai.objective = minimize(L_kakai) # 目的関数
        UELB_kakai += (-L_kakai) >= -1 # 負荷率1以下
        print("容量制限")
        for e in range(len(self.G.edges())): #容量制限
            UELB_kakai += 0 <= L_kakai - ((xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for l in self.G.all_flows])) / self.capacity[self.r_kakai[e][1]])
        print("フロー保存則1")
        for l in self.G.all_flows: #フロー保存則
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == l.get_s()]) == l.get_demand()
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == l.get_s()]) == 0
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == l.get_t()]) == 0
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == l.get_t()]) == l.get_demand()
        print("フロー保存則2")
        for l in self.G.all_flows: #フロー保存則
            for v in self.G.nodes():
                if(v != l.get_s() and v != l.get_t()):
                    UELB_kakai += xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == v])\
                    ==xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == v])

        print("start optimize")
        #線形計画問題を解く
        start = time.time()
        UELB_kakai.optimize()
        elapsed_time = time.time()-start
        status = UELB_kakai.status

        with open('exactsolution.csv', 'a', newline='') as f:
            out = csv.writer(f)
            out.writerow([self.episode, UELB_kakai.objective_value, elapsed_time]) 
        return UELB_kakai.objective_value,elapsed_time