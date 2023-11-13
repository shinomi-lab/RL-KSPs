# coding: utf-8
# import networkx as nx
class Flow():
    def __init__(self,G,flow_id,s,t,demand):
        self.G = G # graph
        self.flow_id = flow_id # フロー番号
        self.s = s # 始点s
        self.t = t # 終点t
        self.demand = demand # 需要量
        self.paths = [] # sからtまでのパス
        self.edges = [] # 流れている辺の集合

        self.area_id = None # フローがエリアフローとして属しているエリアのエリア番号
        self.area_flow_id = None # エリア内でのフロー番号
        self.area_path = [] # エリアグラフにおけるパス

        candidate_s_area = [] # s の存在する管理領域
        if(s in G.nodes()):
            candidate_s_area.append(0)
        self.s_area = list(candidate_s_area)
    
        candidate_t_area = [] # t の存在する管理領域
        if(t in G.nodes()):
            candidate_t_area.append(0)
        self.t_area = list(candidate_t_area)

        self.update_s = s # 更新された始点s
        self.update_t = t # 更新された終点t
        self.up_paths = [] # update_sからupdate_tまでのパス

        self.update_s_area = list(candidate_s_area) #更新後の始点update_sが存在するエリア
        self.update_t_area = list(candidate_t_area) #更新後の終点update_tが存在するエリア    

        self.passing_area = [] # 既に通ったエリア（s; t が同じ管理領域内に存在しないとき）

        # super(Flow, self).__init__()

    def get_id(self): #idの取得
        return self.flow_id
    def get_s(self): #始点の取得
        return self.s
    def get_t(self): #終点の取得
        return self.t

    ###  始点終点の更新関連
    def get_update_s(self): #更新後の始点の取得
        return self.update_s
    def get_update_t(self): #更新後の終点の取得
        return self.update_t
    def set_update_s(self,update_s): #始点の更新
        self.update_s = update_s
        # update_sの存在する管理領域を設定
        candidate_s_area = []
        if(update_s in self.G.nodes()):
            candidate_s_area.append(0)
        self.update_s_area = list(candidate_s_area)
    def set_update_t(self,update_t): #終点の更新
        self.update_t = update_t
        # update_sの存在する管理領域を設定
        candidate_t_area = []
        if(update_t in self.G.nodes()):
            candidate_t_area.append(0)
        self.update_t_area = list(candidate_t_area)
    
    ### 需要量関連
    def get_demand(self): #需要量の取得
        return self.demand
    def set_demand(self,demand): #需要量の設定
        self.demand = demand

    ### パス関連
    def set_paths(self,paths): #パスの設定
        self.paths = paths
    def get_paths(self): #パスの取得
        return self.paths
    def set_up_paths(self,up_paths): # update_sからupdate_tまでのパスの設定
        self.up_paths = up_paths
    def get_up_paths(self): # update_sからupdate_tまでのパスの取得
        return self.up_paths
    def clear_up_paths(self): # update_sからupdate_tまでのパスリストの初期化
        self.up_paths.clear()\
    
    def get_edges(self): #フローが流れている辺の集合の取得
        return self.edges
    def append_edge(self,source,target): #フローが流れている辺の集合に新たな辺を追加
        self.edges.append(tuple(source,target))

    ### エリア関連
    def get_area_id(self): # area_idの取得
        return self.area_id
    def set_area_id(self,area_id): # area_idの設定
        self.area_id = area_id
    def get_s_area(self): #始点sが存在するエリアを取得
        return self.s_area
    def get_t_area(self): #終点tが存在するエリアを取得
        return self.s_area
    def get_update_s_area(self): #更新後の始点update_sが存在するエリアを取得
        return self.update_s_area
    def get_update_t_area(self): #更新後の終点update_tが存在するエリアを取得
        return self.update_t_area

    ### エリア×パス関連
    def get_area_path(self): #エリアパスの取得
        return self.area_path
    def set_area_path(self,area_path): #エリアパスの設定
        self.area_path = area_path
    def get_passing_area(self): #通過したエリアの取得
        return self.passing_area
    def append_passing_area(self,area): #通過したエリアの追加
        self.passing_area.append(area)
    def get_area_flow_id(self): #エリアにおけるフローIDの取得
        return self.area_flow_id
    def set_area_flow_id(self,area_flow_id): #エリアにおけるフローIDの設定
        self.area_flow_id = area_flow_id