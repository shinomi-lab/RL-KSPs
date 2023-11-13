from mip import *
import time
import csv
import networkx as nx
from flow import Flow
import random
import matplotlib.pyplot as plt
import graph_making


def gridgraph(n, m): # gridgraph生成
    G = nx.grid_2d_graph(n, m) 
    n = n*m
    mapping = dict(zip(G, range(1, 100)))
    G = nx.relabel_nodes(G, mapping)
    G = nx.DiGraph(G) # 有向グラフに変換
    for i, j in G.edges(): # capaciy設定
        random.seed()
        v = random.randrange(300, 1000)
        G.adj[i][j]['capacity'] = v
    G.number_of_area = n*m
    G.area_nodes_dict = dict()
    G.flow_initialValues = dict()
    G.all_flows = list()
    for a in range(n*m):
        G.area_nodes_dict[a] = []
    for i, j in G.edges():
        G.add_edge(i, j, flow = dict(G.flow_initialValues), x = dict(), x_kakai = dict())
    pos = nx.spring_layout(G)
    capacity = nx.get_edge_attributes(G, 'capacity')
    edge_list = list(enumerate(G.edges()))
    edges_notindex = []
    for i in range(len(edge_list)):
        edges_notindex.append(edge_list[i][1])
    return G, pos, capacity, edge_list, edges_notindex 

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
    for i, j in G.edges(): #capaciy設定
        random.seed()
        v = random.randrange(100, 10000)
        G.adj[i][j]['capacity'] = v
    G.number_of_area = n
    G.area_nodes_dict = dict()
    G.flow_initialValues = dict()
    G.all_flows = list()
    for a in range(n):
        G.area_nodes_dict[a] = []
    for i, j in G.edges():
        G.add_edge(i, j, flow = dict(G.flow_initialValues), x = dict(), x_kakai = dict())
    pos = nx.circular_layout(G)
    capacity = nx.get_edge_attributes(G, 'capacity')
    edge_list = list(enumerate(G.edges()))
    edges_notindex = []
    for i in range(len(edge_list)):
        edges_notindex.append(edge_list[i][1])
    return G, pos, capacity, edge_list, edges_notindex

def generate_commodity(G,commodity): # 品種の定義(numpy使用)
    determin_st = []
    commodity_dictlist = []
    commodity_list = []
    for i in range(commodity): # commodity generate
        commodity_dict = {}
        s , t = tuple(random.sample(G.nodes, 2)) # source，sink定義
        demand = random.randint(10, 1000) # demand設定
        tentative_st = [s,t] 
        while True:
            if tentative_st in determin_st:
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
        # commodity_list.append(commodity_dict)

        commodity_list.append([s,t,demand])
        commodity_dictlist.append(commodity_dict)
        # print("commodity_dict",commodity_dict)
        # print("commodity_list",commodity_list)
    commodity_list.sort(key=lambda x: -x[2]) # demand大きいものから降順

    return commodity_dictlist,commodity_list  

def debaggraph(n,m): # デバッグ用
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
    G.number_of_area = n*m
    G.area_nodes_dict = dict()
    G.flow_initialValues = dict()
    G.all_flows = list()
    for a in range(n*m):
        G.area_nodes_dict[a] = []
    for i, j in G.edges():
        G.add_edge(i, j, flow = dict(G.flow_initialValues), x = dict(), x_kakai = dict())
    return G

with open('exactsolution_test.csv','w') as f:
    out = csv.writer(f)

random_p = 0.15 # randomgraphの密度
node_l = 3 # gridgraphの列数範囲
node_h = 3 # gridgraphの列数範囲
capa_l = 100 # capacityの範囲
capa_h = 10000 # capacityの範囲

range_commodity_l = 2 # 品種の範囲
range_commodity_h = 2 # 品種の範囲

debag = 0
original = 0
graph_model = "random"
random.seed(1) #ランダムの固定化

for i in range(1):
    commodity = random.randint(range_commodity_l, range_commodity_h) # ランダムで品種数を決定
    node = random.randint(node_l, node_h) # ランダムでノード数を決定
    if original == 1:
        # gridgraph
        G, pos, capaity_list, edge_list, edges_notindex =  gridgraph(node,node) 
        # randomgraph
        # G, pos, capacity_list, edge_list, edges_notindex =  randomgraph(node, random_p)
        
        commodity_dictlist, commodity_list = generate_commodity(G,commodity) # 品種作成
            # デバッグ用
        if debag == 1:
            commodity = 2
            G = debaggraph(2,2)
            capacity_list = {(1, 3): 600, (1, 2): 700, (2, 1): 700, (2, 4): 400, (3, 1): 700, (3, 4): 800, (4, 2): 800, (4, 3): 900}
            edge_list = [(0, (1, 3)), (1, (1, 2)), (2, (2, 1)), (3, (2, 4)), (4, (3, 1)), (5, (3, 4)), (6, (4, 2)), (7, (4, 3))]
            edges_notindex = [(1, 3), (1, 2), (2, 1), (2, 4), (3, 1), (3, 4), (4, 2), (4, 3)]
            commodity_list = [[2, 3, 87], [1, 2, 62]]
            allcommodity_ksps = [[[2, 1, 3], [2, 4, 3]], [[1, 2], [1, 3, 4, 2]]]
            allcommodity_notfstksps = [[[2, 4, 3]], [[1, 3, 4, 2]]]
            combination = [[[2, 1, 3], [1, 2]], [[2, 1, 3], [1, 3, 4, 2]], [[2, 4, 3], [1, 2]], [[2, 4, 3], [1, 3, 4, 2]]]

        r_kakai = list(enumerate(G.edges()))
        commodity_count = 0
        tuples = []
        capacity = nx.get_edge_attributes(G, 'capacity')   

        while(len(tuples)<len(commodity_list)): 
            s = commodity_list[commodity_count][0] # source
            t = commodity_list[commodity_count][1] # sink
            demand = commodity_list[commodity_count][2] # demand
            tuples.append((s,t))
            f = Flow(G,commodity_count,s,t,demand)
            G.all_flows.append(f)
            commodity_count +=1
        
        UELB_kakai = Model('UELB_kakai')#モデルの名前
        L_kakai = UELB_kakai.add_var('L_kakai',lb = 0, ub = 1)
        flow_var_kakai = []
        for l in range(len(G.all_flows)):
            x_kakai = [UELB_kakai.add_var('x{}_{}'.format(l,m), var_type = BINARY) for m, (i,j) in r_kakai]#enumerate関数のmとエッジ(i,j)
            flow_var_kakai.append(x_kakai) #品種エル(l)に対して全ての辺のIDを表す番号mがついている。中身は0-1

        #目的関数
        UELB_kakai.objective = minimize(L_kakai)
        
        #負荷率1以下
        UELB_kakai += (-L_kakai) >= -1
        
        #容量制限
        for e in range(len(G.edges())): 
            UELB_kakai += 0 <= L_kakai - ((xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for l in G.all_flows])) / capacity[r_kakai[e][1]])
        
        #フロー保存則
        for l in G.all_flows:
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][0] == l.get_update_s()]) == l.get_demand()
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][1] == l.get_update_s()]) == 0
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][0] == l.get_update_t()]) == 0
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][1] == l.get_update_t()]) == l.get_demand()
        
        #フロー保存則
        for l in G.all_flows:
            for v in G.nodes():
                if(v != l.get_update_s() and v != l.get_update_t()):
                    UELB_kakai += xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][0] == v])\
                    ==xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][1] == v])

        #線形計画問題を解く
        UELB_kakai.optimize()

        with open('exactsolution_test.csv', 'a', newline='') as f:
            out = csv.writer(f)
            out.writerow([i, UELB_kakai.objective_value]) 
    
        print('Objective :', UELB_kakai.objective_value) #最小の最大負荷率
        print('commodity :', commodity_list) #全ての品種
        print('--------------------------------------------')

        nx.draw(G, with_labels=True)
        plt.show()

    else:
        if(graph_model == 'grid'):
            G = graph_making.Graphs(commodity,commodity) #品種数,areaの品種数　この時点では何もグラフが作られていない　インスタンスの作成？
            G.gridMaker(G,node*node,node,node,0.1,capa_l,capa_h) #G,エリア数,エリアのノード数,列数,行数,ε浮動小数点
            # G.gridMaker(G,1,node*node,node,node,0.1) #G,エリア数,エリアのノード数,列数,行数,ε浮動小数点
            # print(G.area_nodes_dict[0]) # ノードのリスト
            # nx.draw(G)
            # plt.show()
        if(graph_model == 'random'):
            G = graph_making.Graphs(commodity,commodity) #品種数,areaの品種数
            G.randomGraph(G, 2, i, node, capa_l, capa_h) # 5 is each node is joined with its k nearest neighbors in a ring topology. 5はノード数に関係しそう　次数だから
            # G.randomGraph(G, node, 5, i, 1, node, area_height=node_l)
            # nx.draw(G)
            # plt.show()
        print("finish G")
        # print(G.edges())

        r_kakai = list(enumerate(G.edges())) #グラフのエッジと番号を対応付けたもの
        All_commodity_list = []
        commodity_count = 0
        # nodes_list = list(G.area_nodes_dict[0]) # ノードのリスト

        #始点と終点の集合S,Tをつくる
        tuples = []
        # ↓↓↓必要なさそう
        # flows_list = []
        # area_commodity_count = 0
        while(len(tuples)<commodity):
            if(graph_model == 'grid'):
                s, t = random.sample(G.nodes(),2)
                demand  = random.randrange(1, 41, 1)            
            if(graph_model == 'random'):#ノードの中から始点と終点を1つずつランダムで選ぶ
                s, t = random.sample(G.nodes(),2)
                demand  = random.randrange(1, 1000)
            if(s!=t and ((s,t) not in All_commodity_list)): #流し始めのノードからそのノードに戻ってくる組み合わせは考えていない 
                tuples.append((s,t))
                All_commodity_list.append((s,t))
                f = Flow(G,commodity_count,s,t,demand) # インスタンスの作成
                G.all_flows.append(f) # グラフの属性all_flowリストにインスタンスを格納
                # ↓↓↓必要なさそう
                # f.set_area_flow_id(area_commodity_count) # 品種番号設定関数の呼び出し
                # flows_list.append(f) # インスタンスをリストへ格納＝格品種のインスタンスが格納
                commodity_count +=1
                # ↓↓↓必要なさそう
                # area_commodity_count += 1
        # ↓↓↓必要なさそう
        # G.all_flow_dict[0] = flows_list
        print("finish commodity")

        # 問題の定義(全体の下界)
        UELB_kakai = Model('UELB_kakai') # モデルの名前

        L_kakai = UELB_kakai.add_var('L_kakai',lb = 0, ub = 1)
        flow_var_kakai = []
        for l in range(len(G.all_flows)):
            x_kakai = [UELB_kakai.add_var('x{}_{}'.format(l,m), var_type = BINARY) for m, (i,j) in r_kakai]#enumerate関数のmとエッジ(i,j)
            flow_var_kakai.append(x_kakai) #品種エル(l)に対して全ての辺のIDを表す番号mがついている。中身は0-1

        UELB_kakai.objective = minimize(L_kakai) # 目的関数

        UELB_kakai += (-L_kakai) >= -1 # 負荷率1以下
        capacity = nx.get_edge_attributes(G,'capacity')#全辺のcapacityの値を辞書で取得

        # break

        for e in range(len(G.edges())): #容量制限
            UELB_kakai += 0 <= L_kakai - ((xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for l in G.all_flows])) / capacity[r_kakai[e][1]])
        for l in G.all_flows: #フロー保存則
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][0] == l.get_update_s()]) == l.get_demand()
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][1] == l.get_update_s()]) == 0
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][0] == l.get_update_t()]) == 0
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][1] == l.get_update_t()]) == l.get_demand()
        for l in G.all_flows: #フロー保存則
            for v in G.nodes():
                if(v != l.get_update_s() and v != l.get_update_t()):
                    UELB_kakai += xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][0] == v])\
                    ==xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(G.edges())) if r_kakai[e][1][1] == v])

        print("start optimize")
        #線形計画問題を解く
        UELB_kakai.optimize()

        with open('exactsolution_test.csv', 'a', newline='') as f:
            out = csv.writer(f)
            out.writerow([i, UELB_kakai.objective_value]) 
    
        print('Objective :', UELB_kakai.objective_value) #最小の最大負荷率
        print('commodity :', All_commodity_list) #全ての品種
        print('--------------------------------------------')

        # print(nx.get_edge_attributes(G, 'capacity'))
        # print("")
        # print(nx.get_edge_attributes(G, 'update_capacity'))
        # print("")
        # print(nx.get_edge_attributes(G, 'length'))
        # print("")
        # print(nx.get_edge_attributes(G, 'flow_list'))
        # print("")
        # print(nx.get_edge_attributes(G, 'flow_demand_init'))
        # print("")
        # print(nx.get_edge_attributes(G, 'flow_init'))
        # print("")
        # print(nx.get_edge_attributes(G, 'flow_kakai'))
        # print("")
        # print(nx.get_edge_attributes(G, 'flow_kakai_donyoku'))
        # print("")
        # print(nx.get_edge_attributes(G, 'flow'))
        # print("")
        # print(nx.get_edge_attributes(G, 'flow_frac'))
        # print("")
        # print(nx.get_edge_attributes(G, 'flow_frac_donyoku'))
        # print("")
        # print(nx.get_edge_attributes(G, 'elb_flow'))
        # print("")
        # print(nx.get_edge_attributes(G, 'candidate_flows'))
        # print("")
        # print(nx.get_edge_attributes(G, 'load_factor'))
        # print("")
        # print(nx.get_edge_attributes(G, 'load_factor_init'))
        # print("")
        # print(nx.get_edge_attributes(G, 'load_factor_frac'))
        # print("")
        # print(nx.get_edge_attributes(G, 'flow_init'))
        # print("")
        # print(nx.get_edge_attributes(G, 'flow_kakai'))
        # print("")
        # print(nx.get_edge_attributes(G, 'flow_kakai_donyoku'))
        # print("")
        # print(nx.get_edge_attributes(G, 'load_factor_part'))
        # print("")
        # print(nx.get_edge_attributes(G, 'x'))
        # print("")
        # print(nx.get_edge_attributes(G, 'x_kakai'))
        # print("")
        # print(nx.get_edge_attributes(G, 'x_donyoku_kakai'))
        # print("")
        # print(nx.get_edge_attributes(G, 'xf'))
        # print("")
        # print(nx.get_edge_attributes(G, 'xf_donyoku'))
        # print("")
        # print(nx.get_edge_attributes(G, 'x_init'))
        # print("")
        # print(nx.get_edge_attributes(G, 'flag'))
        # print(G.all_flows)

        # nx.draw(G, with_labels=True)
        # plt.show()