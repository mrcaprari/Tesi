from typing import Any, Dict, Optional


def get_meta(arg: Any, key: str, default: Optional[Any] = None) -> Any:
    """
    Safely retrieves a value from the 'meta' attribute of an argument, if it exists.

    Args:
        arg (Any): The object to retrieve metadata from.
        key (str): The key to search for in the metadata.
        default (Optional[Any]): The default value to return if the key or 'meta' does not exist.

    Returns:
        Any: The value associated with the key, or the default value if not found.
    """
    return arg.meta.get(key, default) if hasattr(arg, "meta") else default


def gaussian_mean_node(
    new_node: Any, old_node: Any, n_samples_node: Any, val_map: Dict[Any, Any]
) -> Any:
    """
    Updates a node to point to the 'mu' attribute for Gaussian mean transformations.

    Args:
        new_node (Any): The new node being transformed.
        old_node (Any): The original node in the graph.
        n_samples_node (Any): The node representing 'n_samples' (unused here).
        val_map (Dict[Any, Any]): A mapping of original nodes to new nodes (unused here).

    Returns:
        Any: The transformed node.
    """
    if old_node.op == "get_attr":
        new_node.target = f"{old_node.target}.mu"
    return new_node


def random_sample_node(
    new_node: Any, old_node: Any, n_samples_node: Any, val_map: Dict[Any, Any]
) -> Any:
    """
    Updates a node to represent a random sampling transformation.

    Args:
        new_node (Any): The new node being transformed.
        old_node (Any): The original node in the graph.
        n_samples_node (Any): The node representing 'n_samples'.
        val_map (Dict[Any, Any]): A mapping of original nodes to new nodes.

    Returns:
        Any: The transformed node.
    """
    if old_node.op == "get_attr":
        new_node.op = "call_module"
        new_node.args = (val_map[n_samples_node],)

    if old_node.op == "call_function":
        # Update arguments and set in_dims for transformed arguments
        new_node.args = tuple(val_map.get(arg, arg) for arg in old_node.args)
        in_dims = tuple(
            0 if get_meta(arg, "transform", False) else None for arg in new_node.args
        )
        new_node.meta = {**old_node.meta, "in_dims": in_dims}

    return new_node
