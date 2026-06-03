from furu import furu_method


@furu_method
def skibidi(a, b: int) -> int:
    return a + b


if __name__ == "__main__":
    res = skibidi(a=33, b=34)
    obj = skibidi.furu(a=33, b=34)
    __import__("IPython").embed(header="")
