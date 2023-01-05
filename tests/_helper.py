import rich


def debug_logging(*params, name: str = "\t $$ PyTEST Logging: "):
    rich.print(name, *params)