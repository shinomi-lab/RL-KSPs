# solverを使用しない厳密解

K = 2 #パスの個数
k = 1  #kの値
sumtime = 0 #処理時間
rate = 0 #approximate rate
min_maxload = 1
ks_min_maxload = 0
r_kakai = list(enumerate(g.edges()))
rr_kakai = []
for i in range(len(r_kakai)):
    rr_kakai.append(r_kakai[i][1])

while k < K+1:
    #### 組み合わせを求めるための処理時間計測 ####
    conbination_s = time.time()

    z = 0
    result = []
    conbination = []
    listpath = []
    if k==1:
        for z in range(len(variety)):
            kpath = [] 
            #### Yen's algorithmの処理時間計測 ####
            # path_s = time.time()

            #### Yen's algorithm ####
            X = nx.shortest_simple_paths(g, variety[z][0], variety[z][1]) 
            for counter, path in enumerate(X):
                kpath.append(path) 
                if counter == k-1: 
                    break
            
            #### Yen's algorithmの終了時間計測 ####
            # path_e = time.time()
            # pathsum = path_e-path_s
            result.append(kpath)

        #### 組み合わせを求める ####
        q = [*product(*result)] 
        for conbi in q:
            conbination.append(conbi)
        # print("conbination",conbination)
        # print(len(conbination))
    else:
        for z in range(len(variety)):
            kpath=[]
            #### Yen's algorithmの処理時間計測 ####
            # path_s=time.time()

            #### Yen's algorithm ####
            X = nx.shortest_simple_paths(g, variety[z][0], variety[z][1])
            for counter, path in enumerate(X):
                kpath.append(path)
                if counter == k-1: 
                    listpath.append(path)
                    break
            
            #### Yen's algorithmの終了時間計測 ####
            # path_e = time.time()
            # pathsum =+ path_e-path_s
            result.append(kpath)

        if len(listpath) == 0:
            print("no more conbination")
            break 

        #### 被りの組み合わせを排除 ####
        a = 0
        b = 0
        while a <= len(result)-1:
            b = 0
            while b <= len(listpath)-1:
                if result[a][0][0] == listpath[b][0] and result[a][0][len(result[a][0])-1] == listpath[b][len(listpath[b])-1]:
                    del result[a][len(result[a])-1]
                    copyresult = copy.copy(result)
                    del copyresult[a]
                    addpath = []
                    addpath.append(listpath[b])
                    copyresult.insert(a, addpath)

                    #### 組み合わせを求める ####
                    q = [*product(*copyresult)]
                    for conbi in q:
                        conbination.append(conbi)

                b = b+1
            a = a+1
    # print(conbination)

    #### 組み合わせを求めるための計算終了時間計測 ####
    conbination_e = time.time()
    sumtime = sumtime + (conbination_e-conbination_s)

    #### 経路をもとに辺の01変換処理 ####
    All_conbination = []
    for c in range(len(conbination)):
        flow_var_kakai = []
        for l in g.all_flows:
            x_kakai = len(g.edges())*[0]
            for a in range(len(conbination[c][l.get_id()])):
                if a == len(conbination[c][l.get_id()])-1:
                    break
                set = (conbination[c][l.get_id()][a],conbination[c][l.get_id()][a+1])
                idx = rr_kakai.index(set)
                # print(set)
                x_kakai[idx] = 1
            flow_var_kakai.append(x_kakai)
        All_conbination.append(flow_var_kakai)
    # print(All_conbination)

    Maxload = np.empty(0)
    for c in range(len(All_conbination)):
        load=[]
        #### 問題の定義 ####
        Kshortestpath = Model('Kshortestpath') #モデルの名前

        #### 変数Lの生成 ####
        L = Kshortestpath.add_var('L',lb = 0, ub = 1)

        #### 目的関数 ####
        Kshortestpath.objective = minimize(L)

        #### 制約式 ####
        Kshortestpath += L <= 1 #負荷率1以下
        for e in range(len(g.edges())): #容量制限
            Kshortestpath += 0 <= L - ((xsum([(All_conbination[c][l.get_id()][e])*(l.get_demand()) for l in g.all_flows])) / capacity[r_kakai[e][1]])
            load.append(sum((All_conbination[c][l.get_id()][e])*(l.get_demand())for l in g.all_flows)/capacity[r_kakai[e][1]])
        
        #### 最大負荷率計算処理時間の計測 ####
        rate_s = time.time()

        #### 問題を解く ####
        Kshortestpath.optimize() #L.x == Kshortestpath.objective_value
        Maxload = np.append(Maxload,Kshortestpath.objective_value)

        #### 最大負荷率計算処理の終了時間計測 ####
        rate_e = time.time()
        sumtime = sumtime + (rate_e - rate_s)

    #### K=kにおける近似解計算処理時間計測 ####   
    rate_s = time.time()

    #### K=kにおける近似解の導出 ####
    ks_min_maxload = Maxload.min()
    if min_maxload > ks_min_maxload:
        min_maxload = ks_min_maxload

    #### K=kにおける近似解計算終了時間計測 ####
    rate_e = time.time()
    sumtime = sumtime + (rate_e - rate_s)

    #### 近似率計算 ####
    if UELB_kakai.objective_value == None:
        rate=0
    else:
        rate = float(UELB_kakai.objective_value) / min_maxload 

    with open('/Users/takahashihimeno/Documents/result/approximatesolution-detail.csv', 'a', newline='') as f:
        out = csv.writer(f)
        if testcounter == 1 and k == 1:
            out.writerow(['testcounter','k','min-maxload(Yen)','time'])
        out.writerow([testcounter, k, min_maxload, sumtime])
    
    k = k+1
    #### 計算終了条件 ####
    if k == K+1: #設定値Kまで終了
        print("all K are finished")
    # if rate == 1.0: #近似率100％の近似解計算終了
    #     print("approximate rate is 100%")
    #     break
    # if rate >= 0.8: #近似率80％以上の近似解計算終了
    #     print("approximate rate is 80% over")
    #     break
    if rate == 0: #厳密解が存在しない
        rate='NULL'
        print("exact solution is None")
        break

#### 結果 ####
with open('/Users/takahashihimeno/Documentsapproximatesolution.csv', 'a', newline='') as f:
    out = csv.writer(f)
    if testcounter==1:
        out.writerow(['testcounter','k','min-maxload','sumtime','rate'])
    out.writerow([testcounter, k-1, min_maxload, sumtime, rate])
# print("-------------近似解-------------")
# print("品種：", variety)
# print("最終kの値", k-1)
# print("最小の最大負荷率", min_maxload)
# print("計算時間:", sumtime)
# print("近似率", rate)
