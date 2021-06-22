def load_multi_list(filepath: str, sep="\t"):
    item_list = []
    with open(filepath, "r") as f:
        for item in f:
            tmp_list = item.rstrip().split(sep)
            item_list.append(tmp_list)

    return item_list
