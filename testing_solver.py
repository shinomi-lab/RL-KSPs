# MIP_UELBの不要部分を削除している作成中プログラムファイル

from mip import *
# from pulp import *
import pulp
import time
import csv
import networkx as nx
from flow import Flow
import random
import matplotlib.pyplot as plt
import graph_making
from pyscipopt import Model as SCIPModel, quicksum

with open('exactsolution_test.csv','w') as f:
    out = csv.writer(f)

random_p = 0.15 # randomgraphの密度
node_l = 3 # gridgraphの列数範囲
node_h = 3 # gridgraphの列数範囲
capa_l = 1000 # capacityの範囲
capa_h = 10000 # capacityの範囲
demand_l = 1
demand_h = 500
degree = 2
range_commodity_l = 2 # 品種の範囲
range_commodity_h = 2 # 品種の範囲
graph_model = "random"
random.seed(1) #ランダムの固定化
solver_type = "SCIP"

for i in range(1):
    commodity = random.randint(range_commodity_l,range_commodity_h) # ランダムで品種数を決定
    node = random.randint(node_l, node_h) # ランダムでノード数を決定

    if(graph_model == 'grid'):
        G = graph_making.Graphs(commodity) #品種数,areaの品種数　この時点では何もグラフが作られていない　インスタンスの作成？
        G.gridMaker(G,node,node,0.1,capa_l,capa_h) #G,エリア数,エリアのノード数,列数,行数,ε浮動小数点
    if(graph_model == 'random'):
        G = graph_making.Graphs(commodity) #品種数,areaの品種数
        # G.randomGraph(G, degree, i, node, capa_l, capa_h) # 5 is each node is joined with its k nearest neighbors in a ring topology. 5はノード数に関係しそう　次数だから
        G.randomGraph(G, degree, node, capa_l, capa_h) # 5 is each node is joined with its k nearest neighbors in a ring topology. 5はノード数に関係しそう　次数だから

    print("finish G")
    r_kakai = list(enumerate(G.edges())) #グラフのエッジと番号を対応付けたもの
    All_commodity_list = []
    Commodity_list = []
    commodity_count = 0

    #始点と終点の集合S,Tをつくる
    tuples = []
    while(len(tuples)<commodity):
        s, t = random.sample(G.nodes(),2)
        demand  = random.randrange(demand_l, demand_h)            
        if(s!=t and ((s,t) not in All_commodity_list)): #流し始めのノードからそのノードに戻ってくる組み合わせは考えていない 
            tuples.append((s,t))
            All_commodity_list.append((s,t))
            Commodity_list.append((s,t,demand))
            f = Flow(G,commodity_count,s,t,demand) # インスタンスの作成
            G.all_flows.append(f) # グラフの属性all_flowリストにインスタンスを格納
            commodity_count +=1
    print("finish commodity")
    capacity = nx.get_edge_attributes(G,'capacity') # 全辺のcapacityの値を辞書で取得

    if (solver_type == 'mip'): # mip+CBC
        # 問題の定義(全体の下界)
        UELB_kakai = Model('UELB_kakai') # モデルの名前

        L_kakai = UELB_kakai.add_var('L_kakai',lb = 0, ub = 1)
        flow_var_kakai = []
        for l in range(len(G.all_flows)):
            x_kakai = [UELB_kakai.add_var('x{}_{}'.format(l,m), var_type = BINARY) for m, (i,j) in r_kakai]#enumerate関数のmとエッジ(i,j)
            flow_var_kakai.append(x_kakai) #品種エル(l)に対して全ての辺のIDを表す番号mがついている。中身は0-1

        UELB_kakai.objective = minimize(L_kakai) # 目的関数

        UELB_kakai += (-L_kakai) >= -1 # 負荷率1以下
        
        print("容量制限")
        for e in range(len(G.edges())): #容量制限
            UELB_kakai += 0 <= L_kakai - ((xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for l in G.all_flows])) / capacity[r_kakai[e][1]])
        print("フロー保存則1")
        for l in G.all_flows: #フロー保存則
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][0] == l.get_s()]) == l.get_demand()
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][1] == l.get_s()]) == 0
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][0] == l.get_t()]) == 0
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][1] == l.get_t()]) == l.get_demand()
        print("フロー保存則2")
        for l in G.all_flows: #フロー保存則
            for v in G.nodes():
                if(v != l.get_s() and v != l.get_t()):
                    UELB_kakai += xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][0] == v])\
                    ==xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][1] == v])

        print("start optimize")
        #線形計画問題を解く
        start = time.time()
        UELB_kakai.optimize()
        elapsed_time = time.time()-start

        with open('exactsolution_test.csv', 'a', newline='') as f:
            out = csv.writer(f)
            out.writerow([i, UELB_kakai.objective_value, elapsed_time]) 
    
        print('Objective :', UELB_kakai.objective_value) #最小の最大負荷率
        print('time :', elapsed_time)
        print('commodity :', Commodity_list) #全ての品種
        print('capacity :',capacity)
        # nx.draw(G, with_labels=True)
        # plt.show()
        print('--------------------------------------------')
        # solver_type = "PySCIPOpt"
    
    if (solver_type == 'pulp'): # pulp+CBC
        UELB_problem = pulp.LpProblem('UELB', pulp.LpMinimize) # モデルの名前
        L = pulp.LpVariable('L', 0, 1, 'Continuous') # 最大負荷率　Continuous：連続値 Integer:整数値 Binary:２値変数
        flow_var_kakai = []
        for l in range(len(G.all_flows)): # 0,1変数
            e_01 = [pulp.LpVariable('x{}_{}'.format(l,m), cat=pulp.LpBinary) for m, (i,j) in r_kakai]#enumerate関数のmとエッジ(i,j)
            flow_var_kakai.append(e_01) #品種エル(l)に対して全ての辺のIDを表す番号mがついている。中身は0-1

        UELB_problem += ( L , "Objective" ) # 目的関数値
        
        UELB_problem += (-L) >= -1 # 負荷率1以下

        # print("容量制限")
        for e in range(len(G.edges())): # 容量制限
            UELB_problem += 0 <= L - ((sum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for l in G.all_flows])) / capacity[r_kakai[e][1]])

        # print("フロー保存則1")
        for l in G.all_flows: #フロー保存則
            UELB_problem += sum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][0] == l.get_s()]) == l.get_demand()
            UELB_problem += sum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][1] == l.get_s()]) == 0
            UELB_problem += sum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][0] == l.get_t()]) == 0
            UELB_problem += sum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][1] == l.get_t()]) == l.get_demand()
        
        # print("フロー保存則2")
        for l in G.all_flows: #フロー保存則
            for v in G.nodes():
                if(v != l.get_s() and v != l.get_t()):
                    UELB_problem += sum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][0] == v])\
                    ==sum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][1] == v])

        print("start optimize")
        start = time.time()
        status = UELB_problem.solve() # 線形計画問題を解く
        elapsed_time = time.time()-start

        # print(status)
        print(UELB_problem) # 制約式を全て出してくれる 
        
        with open('exactsolution_test.csv', 'a', newline='') as f:
            out = csv.writer(f)
            out.writerow([i, L.value(),elapsed_time]) 
    
        print('Objective :', pulp.value(UELB_problem.objective))
        print('time :', elapsed_time)
        print('commodity :', Commodity_list) #全ての品種
        print('capacity :',capacity)
        # print(r_kakai)
        # nx.draw(G, with_labels=True)
        # plt.show()
        print('--------------------------------------------')
        # solver_type = "PySCIPOpt"

    if (solver_type == 'SCIP'): # PySCIPOpt+SCIP

        # SCIP Modelの作成
        model = SCIPModel("UELB_problem_SCIP")

        # 変数の定義
        L = model.addVar(vtype="C", name="L", lb=0, ub=1)
   
        flow_var_kakai = []
        for l in range(len(G.all_flows)):
            e_01 = [model.addVar('x{}_{}'.format(l, m), vtype='B') for m, (i, j) in enumerate(r_kakai)]
            flow_var_kakai.append(e_01)

        model.setObjective(L, "minimize")

        model.addCons((L) <= 1)# 負荷率1以下

        for e in range(len(G.edges())): # 容量制限
            model.addCons(L - (quicksum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for l in G.all_flows]) / capacity[r_kakai[e][1]]) >= 0)

        for l in G.all_flows: #フロー保存則
            model.addCons( quicksum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][0] == l.get_s()]) == l.get_demand() )
            model.addCons( quicksum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][1] == l.get_s()]) == 0 )
            model.addCons( quicksum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][0] == l.get_t()]) == 0 )
            model.addCons( quicksum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][1] == l.get_t()]) == l.get_demand() )
        
        for l in G.all_flows: #フロー保存則
            for v in G.nodes():
                if(v != l.get_s() and v != l.get_t()):
                    model.addCons( quicksum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][0] == v])\
                    ==quicksum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][1] == v]) )

        print("start optimize")
        start = time.time()
        model.optimize()
        elapsed_time = time.time()-start
        
        with open('exactsolution_test.csv', 'a', newline='') as f:
            out = csv.writer(f)
            out.writerow([i, -model.getObjVal(),elapsed_time]) 
    
        print('Objective :', model.getObjVal())
        print('time :', elapsed_time)
        print('commodity :', Commodity_list) #全ての品種
        print('capacity :',capacity)
        # print(r_kakai)
        # nx.draw(G, with_labels=True)
        # plt.show()
        print('--------------------------------------------')
        solver_type = "mip"
