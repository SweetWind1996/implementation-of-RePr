
def get_convlayername(model):
    '''
    获取卷积层的名称

    # 参数
        model： 神经网络模型
    '''
    layername = []
    for i in range(len(model.layers)):
        # 将模型中所有层的名称存入列表
        layername.append(model.layers[i].name) 
        # 将卷积层分离出来
    convlayername = [layername[name] for name in range(len(layername)) if 'conv2d' in layername[name]] 
    return convlayername[1:] # 不包括第一层
