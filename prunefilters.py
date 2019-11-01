def prunefilters(model, convlayername, count=0):
    '''
    裁剪filters

    # 参数
        model: 神经网络模型
        convlayername: 保存所有卷积层(2D)的名称
        count: 用于存储每层filters的起始index
    '''
    convnum = len(convlayername) # 卷积层的个数
    params = [i for i in range(convnum)]
    weight = [i for i in range(convnum)]
    MASK = [i for i in range(convnum)]
    rank = dict() # 初始化存储rank的字典
    drop = []
    index1 = 0
    index2 = 0
    for j in range(convnum):
        # 保存卷积层的权重到一个列表，列表的每个元素是一个数组
        params[j] = model.get_layer(convlayername[j]).get_weights() # 将权重转置后才是正常的数组排列(32,32,3,3)
        weight[j] = params[j][0].T
        filternum = weight[j].shape[0] # 获取每一层filter的个数
        # 初始化一个用于判断正交性的矩阵
        W = np.zeros((weight[j].shape[0], weight[j].shape[2]*weight[j].shape[3]*weight[j].shape[1]), dtype='float32')
        for x in range(filternum):
            # filters是一个列表，它的每一个元素是包含一个卷积层所有filter(1D)的列表
            filter = weight[j][x,:,:,:].flatten()
            filter_length = np.linalg.norm(filter) 
            eps = np.finfo(filter_length.dtype).eps
            filter_length = max([filter_length, eps])
            filter_norm = filter / filter_length # 归一化
            # 将每一层的filters放到矩阵的每一行
            W[x,:] = filter_norm
        # 计算层内正交性
        I = np.identity(filternum)
        P = abs(np.dot(W, W.T) - I)
        O = P.sum(axis=1) / 32 # 计算每行元素之和
        for index, o in enumerate(O):
            rank.update({index+count: o})
        count = filternum + count
    # 对字典进行排序，在所有filters上进行ranking
    ranking = sorted(rank.items(), key=lambda x: x[1]) # ranking为一个列表，其元素是存放键值的元组
    for t in range(int(len(ranking)*0.8), len(ranking)):
        drop.append(ranking[t][0])
    for j in range(convnum):
        MASK[j] = np.ones((weight[j].shape), dtype='float32')
        index2 = weight[j].shape[0] + index1
        for a in drop:
            if a >= index1 and a < index2:
                MASK[j][a-index1,:,:,:] = 0
        index1 = index2
    #     weight[j] = (weight[j] * MASK[j]).T
    # for j in range(convnum):
    #     params[j][0] = weight[j]
    #     model.get_layer(convlayername[j]).set_weights(params[j])
    return MASK, weight, drop, convnum, convlayername
