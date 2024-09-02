# TODO can I train it do do step-by-step solutions?
import random
import re
import math

from safe_eval import safe_eval

random.seed(0)

def solve_expression(expr):
    # print('expr:', expr)

    summations = [i for i, c in enumerate(expr) if c in '+']
    products = [i for i, c in enumerate(expr) if c in '*']

    # print('prefix:', prefix)
    # print('summations:', summations)
    # print('products:', products)

    if len(summations) == 0 and len(products) == 0:
        return expr

    operators = products if len(products) > 0 else summations
    operator = random.choice(operators)

    # print('operator:', operator)

    left_match = re.search(r'(.*?)(\d+)$', expr[:operator])
    right_match = re.search(r'^(\d+)(.*)', expr[operator+1:])
    left_expr = left_match.group(1)
    right_expr = right_match.group(2)
    left_num = int(left_match.group(2))
    right_num = int(right_match.group(1))

    # print(f"left_expr: {left_expr}, left_num: {left_num}, right_num: {right_num}, right_expr: {right_expr}")

    combined_num = str(safe_eval(f"{left_num}{expr[operator]}{right_num}"))
    combined_expr = left_expr + combined_num + right_expr

    return expr + "=" + solve_expression(combined_expr)


def reward_function(text):
    statements = text.replace(' ', '').split('=')
    initial = statements.pop(0)

    solution = safe_eval(initial)

    # print(statements)

    error = 0

    # Verify that each statement is true
    for statement in statements:

        if statement.strip() == "":
            error += math.inf
            break

        try:
            error += abs(solution - safe_eval(statement))
        except Exception as e:
            # print(statement, e)
            error += math.inf
            break

    def count_operators(expr):
        return sum([1 for c in expr if c in '+*'])

    # Verify that each statement has one fewer operator
    for i, statement in enumerate(statements):
        if count_operators(statement) != count_operators(initial) - i - 1:
            error = math.inf
            break

    error = min(error, 500)

    # scale error using a sigmoid function in 0 to 1 range
    return 2 / (1 + math.exp(error/10))

    # Scale error using a linear function in 0 to 1 range
    # reward = 1 - error // 30
    # reward = max(0, reward)
    # reward = min(1, reward)
    # return reward


def generate_expression(max_length=10):
    while True:
        expr = ""
        while len(expr) < max_length:
            expr += str(random.randint(0, 10))
            expr += random.choice("+")  # random.choice("+*")

        # Trim to max length
        expr = expr[:max_length]

        # Remove trailing operator if present
        if expr[-1] in "+*":
            expr = expr[:-1]

        if safe_eval(expr) < 100:
            return expr


if __name__ == "__main__":
    expr = "2*4+67*11+1*2"
    result = solve_expression(expr)
    print(result)
    print(reward_function(result))

    print("invalid expression:", reward_function("6*7*9=!42*9=378"))
    print("wrong answer:", reward_function("6*7*9=42*9=377"))
    print("wrong number of operators:", reward_function("6*7*9=378"))

    for i in range(10):
        gen = generate_expression()
        print("\ninitial:", gen)

        result = solve_expression(gen)
        print(result)
        assert reward_function(result) == 1, f"error: {reward_function(result)}"


