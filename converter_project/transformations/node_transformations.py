def get_meta(arg, key, default=None):
    return arg.meta.get(key, default) if hasattr(arg, 'meta') else default


def gaussian_mean_node(new_node, old_node, n_samples_node, val_map):
    if old_node.op == "get_attr":
        new_node.target = f'{old_node.target}.mu'
    return new_node


def random_sample_node(new_node, old_node, n_samples_node, val_map):
    if old_node.op == "get_attr":
        new_node.op = "call_module"
        new_node.args = (val_map[n_samples_node],)

    if old_node.op == "call_function":
        new_node.args = tuple(val_map.get(arg, arg) for arg in old_node.args)
        in_dims = tuple(
            0 if get_meta(arg, 'transform', False) else None
            for arg in new_node.args
        )
        new_node.meta = {**old_node.meta, 'in_dims': in_dims}
    return new_node
