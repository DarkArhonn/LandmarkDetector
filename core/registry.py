REGISTRY = {}


def register(name: str):
    """
    register name in global register
    :param name:
    :return:
    """
    def do_register(cls):
        if name in REGISTRY:
            raise ValueError(f"Connot register class f{cls.name} under name {name}.")
        REGISTRY[name] = cls
        return cls

    return do_register

