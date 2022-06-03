def index_from_file(path):
    """Load a list of newline-separated integer indices from a file"""
    with open(path, "r") as file:
        indices = [int(l.strip("\n")) for l in file.readlines()]
    return indices
