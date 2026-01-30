# calculator/utils.py
def validate_sets(data):
    if isinstance(data, dict):
        for k, v in data.items():
            if not isinstance(v, set):
                raise TypeError(f"Value for key '{k}' must be a set")
    elif isinstance(data, list):
        for i, s in enumerate(data):
            if not isinstance(s, set):
                raise TypeError(f"Item at index {i} must be a set")
    else:
        raise TypeError("Data must be a list of sets or a dict of set_name -> set")
    return True
