import pickle
from datetime import datetime


def create_pickle(path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, f)


def append_pickle(path: str, data: dict[any]) -> None:
    with open(path, "ab") as f:
        pickle.dump(data, f)


def read_pickle(path: str) -> list[any]:
    data = []
    with open(path, "rb") as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
    return data
