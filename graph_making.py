# coding: utf-8
import networkx as nx
import matplotlib.pyplot as plt
import math
import random
import re
import collections

class Graphs(nx.DiGraph):
    def __init__(self,commodity,area_commodity):
        self.G = nx.DiGraph() # 有向グラフを定義
        self.commodity = commodity #フローの品種数

        self.eps = 0 # 誤差の初期値
        self.delta = 0

        self.flows = dict()
        self.flow_initialValues = dict()
        self.elb_flow_initialValues = dict()
        self.x = dict()
        self.initimal_list = list()
        self.all_flows = list()

        #エリアごとのノードをdictで保存
        # self.area_nodes_dict = dict()

        super(Graphs, self).__init__() # nx.DiGraphkクラスを継承するために必要

    # def gridMaker(self, G, number_of_area, node, area_width, area_height, eps): # G,エリア数 1,ノード数,列数,行数,ε浮動小数点
    def gridMaker(self, G, node, area_width, area_height, eps, capa_l, capa_h): # G,エリア数 1,ノード数,列数,行数,ε浮動小数点
        self.G = G
        self.node = node # ノード数
        self.capa_l = capa_l
        self.capa_h = capa_h
        # self.area_width = area_width # 列数
        # self.area_height = area_height # 行数
        width = int(area_width) # 列数
        height = int(area_height) # 行数
        # self.number_of_area = number_of_area # 1
        # nodes = []
        # edges = []
        # self.area_nodes_dict[0] = [] 
        self.eps = eps # 初期値を設定
        self.delta=(1+self.eps)/(((1+self.eps)*self.node)**(1/self.eps))
        #グラフへノードを追加
        for i in range(1,self.node+1):
            self.G.add_node(i)
            # nodes.append(i)
        # for area_node in range(1,node+1):
        # 	self.area_nodes_dict[0].append(area_node)   	
        #グラフへエッジの追加
        for w in range(1,width+1):
            for h in range(1,height+1):
                if(w==width and h==height):
                    break
                elif(w!=width and h==height):
                    self.G.add_bidirectionaledge(self.G,height*w,height*(w+1),capa_l,capa_h)
                elif(h!=height and w==width):
                    self.G.add_bidirectionaledge(self.G,(w-1)*height+h,(w-1)*height+(h+1),capa_l,capa_h)
                else:
                    self.G.add_bidirectionaledge(self.G,(w-1)*height+h,(w-1)*height+(h+1),capa_l,capa_h)
                    self.G.add_bidirectionaledge(self.G,(w-1)*height+h,w*height+h,capa_l,capa_h)

    # def randomGraph(self, G, n, k, seed, number_of_area, node, area_height):
    def randomGraph(self, G, k, seed, node, capa_l, capa_h):
        # G.randomGraph(G, node, 5, i, 1, node, area_height=node_l)
        self.G = G
        self.node = node # node数
        self.capa_l = capa_l
        self.capa_h = capa_h
        # self.number_of_area = number_of_area # 1
        # self.area_height = area_height
        # height = int(area_height)
        ty = 0 # randomグラフにおいて、s=0だとnewman_watts_strogatz_graphになり、1だと任意のrandomグラフ
        if ty == 0:
            # 一旦NWSを作ってGに当てはめていく
            NWS = nx.newman_watts_strogatz_graph(self.node, k, 0.5, seed=seed) # (node, Each node is joined with its k nearest neighbors in a ring topology, probability, seed)
            # self.area_nodes_dict[0] = [] #リストの中のリスト生成？self.area_nodes_dict[[],[]]
            for i in NWS.nodes():
                self.G.add_node(i)
            for (x, y) in NWS.edges():
                self.G.add_bidirectionaledge(self.G, x, y, self.capa_l,  self.capa_h) #下記の関数呼び出し
            # for area_node in range(1,node+1):
            # 	self.area_nodes_dict[0].append(area_node)   

    def add_bidirectionaledge(self,G,x,y,capa_l,capa_h): # 双方向辺を追加する関数
        cap = random.randrange(capa_l,capa_h) # capacityの定義
        # # x --> y の辺を追加
        # self.G.add_edge(x,y,capacity=int(cap),update_capacity=int(cap),length=float(self.delta),flow_list=list(self.initimal_list),flow_demand_init=dict(self.flow_initialValues),flow_init=list(self.initimal_list),flow_kakai=dict(self.flow_initialValues),flow_kakai_donyoku=dict(self.flow_initialValues),flow=dict(self.flow_initialValues),flow_frac=dict(self.flow_initialValues),flow_frac_donyoku=dict(self.flow_initialValues),elb_flow=dict(self.elb_flow_initialValues),candidate_flows=dict(),load_factor=0,load_factor_init=0,load_factor_frac=0,load_factor_frac_donyoku=0,load_factor_kakai=0,load_factor_kakai_donyoku=0,load_factor_part=0,x=dict(),x_kakai=dict(),x_donyoku_kakai=dict(),xf=dict(),xf_donyoku=dict(),x_init=dict(),flag=False) 
        # # y --> x の辺を追加
        # self.G.add_edge(y,x,capacity=int(cap),update_capacity=int(cap),length=float(self.delta),flow_list=list(self.initimal_list),flow_demand_init=dict(self.flow_initialValues),flow_init=list(self.initimal_list),flow_kakai=dict(self.flow_initialValues),flow_kakai_donyoku=dict(self.flow_initialValues),flow=dict(self.flow_initialValues),flow_frac=dict(self.flow_initialValues),flow_frac_donyoku=dict(self.flow_initialValues),elb_flow=dict(self.elb_flow_initialValues),candidate_flows=dict(),load_factor=0,load_factor_init=0,load_factor_frac=0,load_factor_frac_donyoku=0,load_factor_kakai=0,load_factor_kakai_donyoku=0,load_factor_part=0,x=dict(),x_kakai=dict(),x_donyoku_kakai=dict(),xf=dict(),xf_donyoku=dict(),x_init=dict(),flag=False)
        
        # x --> y の辺を追加
        self.G.add_edge(x,y,capacity=int(cap),update_capacity=int(cap),length=float(self.delta),flow_list=list(self.initimal_list),flow_demand_init=dict(self.flow_initialValues),flow_init=list(self.initimal_list),flow_kakai=dict(self.flow_initialValues),flow_kakai_donyoku=dict(self.flow_initialValues),flow=dict(self.flow_initialValues),flow_frac=dict(self.flow_initialValues),flow_frac_donyoku=dict(self.flow_initialValues),elb_flow=dict(self.elb_flow_initialValues),candidate_flows=dict(),load_factor=0,load_factor_init=0,load_factor_frac=0,load_factor_frac_donyoku=0,load_factor_kakai=0,load_factor_kakai_donyoku=0,load_factor_part=0,x=dict(),x_kakai=dict(),x_donyoku_kakai=dict(),xf=dict(),xf_donyoku=dict(),x_init=dict(),flag=False) 
        
        # y --> x の辺を追加
        self.G.add_edge(y,x,capacity=int(cap),update_capacity=int(cap),length=float(self.delta),flow_list=list(self.initimal_list),flow_demand_init=dict(self.flow_initialValues),flow_init=list(self.initimal_list),flow_kakai=dict(self.flow_initialValues),flow_kakai_donyoku=dict(self.flow_initialValues),flow=dict(self.flow_initialValues),flow_frac=dict(self.flow_initialValues),flow_frac_donyoku=dict(self.flow_initialValues),elb_flow=dict(self.elb_flow_initialValues),candidate_flows=dict(),load_factor=0,load_factor_init=0,load_factor_frac=0,load_factor_frac_donyoku=0,load_factor_kakai=0,load_factor_kakai_donyoku=0,load_factor_part=0,x=dict(),x_kakai=dict(),x_donyoku_kakai=dict(),xf=dict(),xf_donyoku=dict(),x_init=dict(),flag=False)

        # for i, j in G.edges():
        # 	G.add_edge(i, j, flow = dict(self.flow_initialValues), x = dict(), x_kakai = dict())