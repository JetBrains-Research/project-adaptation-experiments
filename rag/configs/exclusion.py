def exclusion(scorer: str, splitter: str, n_grams_max: int) -> bool:
    if splitter == "line_splitter":
        if scorer != "iou" or n_grams_max > 1:
            return True
    else:
        return False
