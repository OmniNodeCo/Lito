"""
Calculator extension for Lito.

Handles simple arithmetic: "what is 2 + 2 * 10", "calculate 15% of 200",
"3^8", "/calc (2+3)*4".
"""

import ast
import operator
import re

NAME = 'calculator'
VERSION = '1.0.0'
DESCRIPTION = ('Evaluates arithmetic expressions like '
               '"what is 2 + 2 * 10" or "15% of 200".')
COMMANDS = {'/calc': 'evaluate a math expression, e.g. /calc (2+3)*4'}

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}

_STRIP_LEAD = re.compile(
    r"^(?:hey\s+|please\s+)?"
    r"(?:what(?:'s|s| is)?|how much(?: is)?)\s+(?:is\s+|are\s+)?"
    r"|(?:hey\s+|please\s+)?(?:can you\s+)?"
    r"(?:calculate|compute|evaluate|calc|solve)\s*",
    re.IGNORECASE)
_STRIP_TAIL = re.compile(
    r"\s*(?:\?+|!+|\.+|thanks.*|please.*|equal(?:s)?\s*\??\s*)$",
    re.IGNORECASE)
_ARITH_CHARS = re.compile(r'^[\d\s+\-*/().,%×÷^xX]+$')
_HAS_OPERATOR = re.compile(r'[+\-*/%^×÷xX]')
_HAS_DIGIT = re.compile(r'\d')


def _extract_expr(text: str) -> str:
    expr = _STRIP_LEAD.sub('', (text or '').strip())
    expr = _STRIP_TAIL.sub('', expr).strip()
    # "15% of 200" -> "(15/100)*200"
    expr = re.sub(r'(\d+(?:\.\d+)?)\s*%\s*of\s+',
                  r'(\1/100)*', expr, flags=re.IGNORECASE)
    # allow x / × / ÷ as symbols
    expr = expr.replace('×', '*').replace('÷', '/')
    if ' x ' in f' {expr.lower()} ':
        expr = re.sub(r'\bx\B?', '*', expr, flags=re.IGNORECASE)
    return expr.strip()


def can_handle(text: str) -> bool:
    expr = _extract_expr(text)
    if not expr or not _ARITH_CHARS.match(expr):
        return False
    return bool(_HAS_OPERATOR.search(expr) and _HAS_DIGIT.search(expr))


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if (isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow):
            right = _eval(node.right)
            if abs(right) > 64:
                raise ValueError('exponent too large')
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError('unsupported expression')


def _calculate(expr: str):
    tree = ast.parse(expr.replace('^', '**'), mode='eval')
    value = _eval(tree)
    if isinstance(value, complex):
        raise ValueError('complex results are not supported')
    return value


def _fmt(value) -> str:
    text = f'{value:,.10g}'
    return text


def handle(text: str, brain) -> str:
    expr = _extract_expr(text)
    if not expr:
        return ''
    try:
        value = _calculate(expr)
    except ZeroDivisionError:
        return "I can't divide by zero! 🫣"
    except Exception:
        return ''
    return f"{expr} = {_fmt(value)} 🧮"


def handle_command(cmd: str, args: str, brain) -> str:
    expr = (args or '').strip().replace('^', '**')
    if not expr:
        return f"Usage: {cmd} <expression>   e.g. {cmd} (2+3)*4"
    try:
        value = _calculate(expr)
        return f"{args.strip()} = {_fmt(value)} 🧮"
    except ZeroDivisionError:
        return "I can't divide by zero! 🫣"
    except Exception as e:
        return f"I couldn't calculate that: {e}"
