import ast
import operator


def safe_eval(expr):
    # Define allowed operators
    operators = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg
    }

    def eval_(node):
        if isinstance(node, ast.Num):  # For Python 3.7 and earlier, use ast.Num instead of ast.Constant
            return node.n
        elif isinstance(node, ast.BinOp):
            return operators[type(node.op)](eval_(node.left), eval_(node.right))
        elif isinstance(node, ast.UnaryOp):
            return operators[type(node.op)](eval_(node.operand))
        else:
            raise TypeError("Unsupported type {}".format(type(node)))

    # Parse the expression into an Abstract Syntax Tree (AST)
    parsed_expr = ast.parse(expr, mode='eval')

    # Evaluate the AST
    return eval_(parsed_expr.body)


# Example usage
if __name__ == '__main__':
    expr = "-4*-5*3"
    result = safe_eval(expr)
    print(result)
