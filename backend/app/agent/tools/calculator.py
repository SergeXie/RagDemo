"""安全计算器工具：数据分析场景的表达式求值。

使用 AST 白名单解析，仅允许数值字面量、四则运算、幂运算、
括号、比较与少量 math 函数 —— 无任意代码执行风险。
"""

from __future__ import annotations

import ast
import math
import operator

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class _CalculatorInput(BaseModel):
    expression: str = Field(
        ...,
        description="数学表达式，例如 '1200 * 0.15 + 300' 或 'sqrt(144) + 10%'",
    )


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pow": pow,
    "min": min,
    "max": max,
    "sum": sum,
    "avg": lambda *xs: sum(xs) / len(xs),
}

_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

_MAX_NODES = 100  # 表达式复杂度上限，防 ReDoS / 栈溢出
_MAX_DEPTH = 50   # 递归深度上限，防止深层括号嵌套


def safe_eval(expression: str) -> float:
    """对数学表达式安全求值。

    Raises:
        ValueError: 表达式不合法或包含不允许的结构。
    """
    expression = expression.strip().rstrip("%")  # 容忍结尾百分号
    if not expression:
        raise ValueError("表达式为空")
    if len(expression) > 500:
        raise ValueError("表达式过长")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"表达式语法错误: {exc.msg}") from exc

    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        raise ValueError("表达式过于复杂")

    def _eval(node: ast.AST, depth: int = 0) -> float:
        if depth > _MAX_DEPTH:
            raise ValueError("表达式嵌套过深")
        if isinstance(node, ast.Expression):
            return _eval(node.body, depth + 1)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return node.value
            raise ValueError(f"不允许的字面量: {node.value!r}")
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _BIN_OPS:
                raise ValueError(f"不允许的运算符: {op_type.__name__}")
            try:
                return _BIN_OPS[op_type](_eval(node.left, depth + 1), _eval(node.right, depth + 1))
            except ZeroDivisionError as exc:
                raise ValueError("除数不能为零") from exc
            except OverflowError as exc:
                raise ValueError("数值溢出") from exc
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _UNARY_OPS:
                raise ValueError(f"不允许的一元运算符: {op_type.__name__}")
            return _UNARY_OPS[op_type](_eval(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
                raise ValueError("仅允许调用内置数学函数")
            args = [_eval(a, depth + 1) for a in node.args]
            try:
                return _FUNCTIONS[node.func.id](*args)
            except ValueError:
                raise
            except Exception as exc:  # noqa: BLE001 - math 域错误等统一转 ValueError
                raise ValueError(f"函数计算错误: {exc}") from exc
        if isinstance(node, ast.Name):
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            raise ValueError(f"未知变量: {node.id}")
        if isinstance(node, ast.Compare):
            # 支持链式比较（如 1 < x <= 5 的常量情形）
            left = _eval(node.left, depth + 1)
            for op, comp in zip(node.ops, node.comparators):
                right = _eval(comp, depth + 1)
                if isinstance(op, ast.Lt):
                    ok = left < right
                elif isinstance(op, ast.LtE):
                    ok = left <= right
                elif isinstance(op, ast.Gt):
                    ok = left > right
                elif isinstance(op, ast.GtE):
                    ok = left >= right
                elif isinstance(op, ast.Eq):
                    ok = left == right
                elif isinstance(op, ast.NotEq):
                    ok = left != right
                else:
                    raise ValueError("不允许的比较运算符")
                if not ok:
                    return 0.0
                left = right
            return 1.0
        raise ValueError(f"不允许的表达式元素: {type(node).__name__}")

    result = _eval(tree)
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ValueError("表达式未产生数值结果")
    if not math.isfinite(result):
        raise ValueError("结果不是有限数值")
    return float(result)


def _calculate(expression: str) -> str:
    try:
        value = safe_eval(expression)
    except Exception as exc:  # noqa: BLE001 - 工具需返回可读错误而非抛出
        return f"计算错误：{exc}"
    # 整数结果不显示小数点
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


calculator_tool = StructuredTool.from_function(
    func=_calculate,
    name="calculator",
    description=(
        "安全的数学计算器。当需要进行数值计算（如报销金额、税率、"
        "统计平均值、百分比换算）时使用。输入为纯数学表达式，"
        "支持 + - * / // % **、括号及 sqrt/log/avg/min/max 等函数。"
    ),
    args_schema=_CalculatorInput,
)
