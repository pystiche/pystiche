def maybe_update(dct, key, val):
    if key not in dct:
        dct[key] = val


def exec_file(file, globals=None, locals=None):
    if globals is None:
        globals = {}
    maybe_update(globals, "__file__", file)
    maybe_update(globals, "__name__", "__main__")

    with open(file, "r") as fh:
        exec(fh.read(), globals, locals)

    return globals, locals
