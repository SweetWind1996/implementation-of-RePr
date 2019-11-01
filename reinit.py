def reinit(model, weight, drop, convnum, convlayername):

    index1 = 0
    index2 = 0
    new_params = [i for i in range(convnum)]
    new_weight = [i for i in range(convnum)]
    for j in range(convnum):
        new_params[j] = model.get_layer(convlayername[j]).get_weights() 
        new_weight[j] = new_params[j][0].T
    stack_new_filters = new_weight[0]
    stack_filters = weight[0]
    filter_index1 = 0
    filter_index2 = 0
    for i in range(len(new_weight)-1):
        next_new_filter = new_weight[i+1]
        next_filter = weight[i+1]
        stack_new_filters = np.vstack((stack_new_filters, next_new_filter))
        stack_filters = np.vstack((stack_filters, next_filter))
    stack_new_filters_flat = np.zeros((stack_new_filters.shape[0], 
        stack_new_filters.shape[1]*stack_new_filters.shape[2]*stack_new_filters.shape[3]), dtype='float32')
    stack_filters_flat = np.zeros((stack_filters.shape[0], 
        stack_filters.shape[1]*stack_filters.shape[2]*stack_filters.shape[3]), dtype='float32')
    for p in range(stack_new_filters.shape[0]):
        stack_new_filters_flat[p] = stack_new_filters[p].flatten()
        stack_filters_flat[p] = stack_filters[p].flatten()
    q = np.zeros((stack_new_filters_flat.shape[0]), dtype='float32')
    tol = None
    reinit = None
    solve = None
    for b in drop:
        Q, R= qr(stack_new_filters_flat.T)
        for k in range(R.shape[0]):
            if np.abs(np.diag(R)[k])==0:
                # print(k)
                reinit = Q.T[k]
                break
        null_space = reinit
        stack_new_filters_flat[b] = null_space
    for filter_in_stack in range(stack_new_filters_flat.shape[0]):
        stack_new_filters[filter_in_stack] = stack_new_filters_flat[filter_in_stack].reshape(
            (stack_new_filters.shape[1], stack_new_filters.shape[2], stack_new_filters.shape[3]))
    for f in range(len(new_weight)):
        filter_index2 = new_weight[f].shape[0] + filter_index1
        new_weight[f] = stack_new_filters[filter_index1:filter_index2,:,:,:]
        filter_index1 = new_weight[f].shape[0]
        new_params[f][0] = new_weight[f].T
        model.get_layer(convlayername[f]).set_weights(new_params[f]) 
