#
# This file is part of Cube Builder.
# Copyright (C) 2022 INPE.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.
#

"""Simple abstraction of Python Interpreter."""

import ast
from typing import Any, Dict

# Type for Python Execution Code Context.
ExecutionContext = Dict[str, Any]


def execute(expression: str, context: dict) -> ExecutionContext:
    """Evaluate a string expression as Python object and execute in Python Interpreter.

    This method allows to execute dynamic expression into a Python Virtual Machine.
    With this, you can generate custom bands based in user-defined values. The `context`
    defines the scope of which values will be available by default.

    TODO: Ensure that non-exported variables (context) can't be executed like `os` to avoid
     internal issues.

     Args:
         expression - String-like python expression
         context - Context loaded variables

    Examples:
        >>> import numpy
        ... # Evaluate half of coastal band using numpy.random
        >>> res = execute('coastalHalf = B1 / 2', context=dict(B1=numpy.random.rand(1000, 1000) * 10000))
        >>> res['coastalHalf']

    Notes:
        You can set loaded variables in `context` and it will be available during code execution.

    Returns:
        Map of context values loaded in memory.
    """
    ast_expression = ast.parse(expression)

    compiled_expression = compile(ast_expression, '<ast>', 'exec')

    exec(compiled_expression, context)

    return context
