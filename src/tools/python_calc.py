"""安全算术工具 python_calc。

- _eval_node: 用 AST 只允许数字和四则等运算，禁止 eval 任意代码
- PythonCalcTool.run: 计算 expression，返回结果字符串
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from src.tools.base import Tool

_BIN_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("只允许数字与 + - * / // % ** 和括号")


class PythonCalcTool(Tool):
    name = "python_calc"
    description = "计算纯算术表达式，例如 (3+5)*2。不要传入变量或函数调用。"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "算术表达式"},
        },
        "required": ["expression"],
    }

    def run(self, **kwargs: Any) -> str:
        expression = str(kwargs.get("expression", "")).strip()
        if not expression:
            return "ERROR: expression 为空"
        tree = ast.parse(expression, mode="eval")
        value = _eval_node(tree)
        if value == int(value):
            return str(int(value))
        return str(value)
