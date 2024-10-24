
def exclusion(scorer, splitter, n_grams_max):
    if splitter == "line_splitter":
        if scorer != "iou" or n_grams_max > 1:
            return True
    else:
        return False