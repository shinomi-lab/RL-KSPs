# coding: utf-8
# import networkx as nx
class Flow():
    def __init__(self,G,flow_id,s,t,demand):
        self.G = G # graph
        self.flow_id = flow_id # フロー番号
        self.s = s # 始点s
        self.t = t # 終点t
        self.demand = demand # 需要量
        self.update_s = s # 更新された始点s
        self.update_t = t # 更新された終点t

    def get_id(self): #idの取得
        return self.flow_id

    ###  始点終点の更新関連
    def get_update_s(self): #更新後の始点の取得
        return self.update_s
    def get_update_t(self): #更新後の終点の取得
        return self.update_t
    
    ### 需要量関連
    def get_demand(self): #需要量の取得
        return self.demand