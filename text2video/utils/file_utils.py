import json
import pickle as pkl


def save(filename, data):
    with open(filename, "wb") as file:
        pkl.dump(data, file)


def load(filename):
    try:
        data = pkl.load(open(filename, "rb"))
    except OSError:
        data = []

    return data


def save_list(file, some_list):
    with open(file, "w") as file_obj:
        for element in some_list:
            file_obj.write("%s\n" % element)


def get_lines(file):
    lines = [line.rstrip("\n") for line in open(file, "r")]
    return lines


def save_json(filename, data):
    with open(filename, "w") as outfile:
        json.dump(data, outfile)


def load_json(filename):
    with open(filename, "r") as json_data:
        data = json.load(json_data)

    return data


def merge_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    result = x.copy()
    result.update(y)
    return result
