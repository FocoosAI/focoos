import re


def get_layers_to_prune(regex_pattern: str, layers_file_path: str) -> list[str]:
    """
    Returns a list of layer names from the given file that match the provided regex pattern.

    Args:
        regex_pattern (str): Regular expression pattern to match layer names.
        layers_file_path (str): Path to the file containing layer names (one per line, as keys before ':').

    Returns:
        list[str]: List of matching layer names.
    """
    pattern = re.compile(regex_pattern)
    matching_layers = []
    with open(layers_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            layer_name = line.split(":", 1)[0].strip()
            if pattern.fullmatch(layer_name) or pattern.search(layer_name):
                matching_layers.append(layer_name)

    # remove suffix ".weight"
    suffix = ".weight"
    matching_layers = [layer.replace(suffix, "") for layer in matching_layers]
    return matching_layers


def load_layers_from_file(layers_file_path: str) -> list[str]:
    with open(layers_file_path, "r") as f:
        return [line.strip() for line in f.readlines()]
