def Mask(model, mask):
    convlayername = get_convlayername(model)
    for i in range(len(convlayername)):
        Params = [i for i in range(len(convlayername))]
        Weight = [i for i in range(len(convlayername))]
        Params[i] = model.get_layer(convlayername[i]).get_weights() 
        Weight[i] = (Params[i][0].T*mask[i]).T
        Params[i][0] = Weight[i]
        model.get_layer(convlayername[i]).set_weights(Params[i])

prune_callback = LambdaCallback(
    on_batch_end=lambda batch,logs: Mask(model, mask)) # callback to use mask 
