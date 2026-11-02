"""安全计算器测试。"""

import math

import pytest

from app.agent.tools.calculator import safe_eval


class TestSafeEval:
    def test_basic_arithmetic(self):
        assert safe_eval("1 + 2 * 3") == 7
        assert safe_eval("(1 + 2) * 3") == 9
        assert safe_eval("10 / 4") == 2.5
        assert safe_eval("10 // 3") == 3
        assert safe_eval("10 % 3") == 1
        assert safe_eval("2 ** 10") == 1024

    def test_unary(self):
        assert safe_eval("-5 + 3") == -2
        assert safe_eval("+5") == 5

    def test_functions(self):
        assert safe_eval("sqrt(144)") == 12
        assert safe_eval("abs(-7)") == 7
        assert safe_eval("avg(2, 4, 6)") == 4
        assert safe_eval("min(3, 1, 2)") == 1
        assert safe_eval("max(3, 1, 2)") == 3

    def test_constants(self):
        assert abs(safe_eval("pi") - math.pi) < 1e-9

    def test_round(self):
        assert safe_eval("round(3.14159, 2)") == 3.14

    def test_complex_expression(self):
        # 模拟报销计算：基础 1200 + 住宿 800*0.9 + 交通 350
        assert safe_eval("1200 + 800 * 0.9 + 350") == 2270

    @pytest.mark.parametrize("bad", [
        "__import__('os').system('ls')",   # 导入
        "open('/etc/passwd')",             # 文件访问
        "lambda: 1",                       # lambda
        "[1,2,3]",                         # 列表
        "{1: 2}",                          # 字典
        "x + 1",                           # 未知变量
        "exec('print(1)')",                # exec
        "a.b",                             # 属性访问
        "",                                # 空
        "1 / 0",                           # 除零 -> inf 被拒
    ])
    def test_rejects_malicious(self, bad):
        with pytest.raises(ValueError):
            safe_eval(bad)

    def test_rejects_too_long(self):
        with pytest.raises(ValueError):
            safe_eval("1" * 600)

    def test_rejects_deep_nesting(self):
        with pytest.raises(ValueError):
            # 连续加法产生 60 层左嵌套 BinOp 节点，触发深度/复杂度上限
            safe_eval("+".join(["1"] * 60))

    def test_tool_wrapper_handles_error(self):
        from app.agent.tools.calculator import _calculate

        assert _calculate("bad expr") .startswith("计算错误")
        assert _calculate("2 + 2") == "4"
        assert _calculate("10 / 4") == "2.5"
