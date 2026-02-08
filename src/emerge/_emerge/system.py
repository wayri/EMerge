
import sys
import os
import ast
import inspect

def called_from_main_function() -> bool:
    """
    Check whether the entry-point script contains
    `if __name__ == "__main__":`
    """
    main_mod = sys.modules.get("__main__")
    main_file = getattr(main_mod, "__file__", None)

    if not main_file:
        return False

    main_file = os.path.realpath(main_file)

    try:
        with open(main_file, "r") as f:
            src = f.read()
    except OSError:
        return False

    try:
        tree = ast.parse(src, filename=main_file)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue

        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        ):
            return True

    return False