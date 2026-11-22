"""Accelerate verified SQLGlot executor operations during TPC-DS stage checks."""

import ast
from itertools import product
import types


_NODES = {
    ast.Constant, ast.Tuple, ast.List, ast.Subscript, ast.Slice, ast.Name, ast.Load,
    ast.Call, ast.keyword, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not,
    ast.UAdd, ast.USub, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
    ast.Mod, ast.Pow, ast.IfExp, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
    ast.Gt, ast.GtE,
}

# These functions do not inspect the caller, change its locals, or depend on time.
_CALLS = frozenset((
    'ABS', 'ADD', 'AND', 'AVG', 'BETWEEN', 'COALESCE', 'CONCAT', 'COUNT', 'DIV',
    'EQ', 'FIRST', 'GT', 'GTE', 'IF', 'IN', 'INT', 'INTDIV', 'LEFT', 'LENGTH',
    'LIKE', 'LOWER', 'LT', 'LTE', 'MAX', 'MIN', 'MOD', 'MUL', 'NEQ', 'NOT',
    'OR', 'ORDERED', 'POW', 'RIGHT', 'ROUND', 'SUB', 'SUM', 'TRIM', 'UPPER',
))


def _fuse(codes):
    """Keep the original path for anything outside the supported expression grammar."""
    parts = []
    for code in codes:
        if type(code) is not types.CodeType:
            return None
        source = code.co_filename
        if len(source) > 2048:
            return None
        try:
            tree = ast.parse(source, mode='eval')
            for node in ast.walk(tree.body):
                if type(node) not in _NODES:
                    return None
                if isinstance(node, ast.Call) and (
                    type(node.func) is not ast.Name or node.func.id not in _CALLS
                ):
                    return None
                if isinstance(node, ast.Name) and node.id not in _CALLS and node.id != 'scope':
                    return None
            # The executor uses the generated expression as co_filename. Require
            # equivalent bytecode before using it as source for the combined program.
            if code != compile(tree, source, 'eval', dont_inherit=True, optimize=2):
                return None
        except (SyntaxError, TypeError, ValueError, RecursionError):
            return None
        parts.append(tree.body)
    fused = ast.Expression(ast.Tuple(elts=parts, ctx=ast.Load()))
    try:
        return compile(ast.fix_missing_locations(fused), '<sqlglot projection tuple>', 'eval', dont_inherit=True, optimize=2)
    except (SyntaxError, TypeError, ValueError, RecursionError):
        return None


class _Projection(tuple):
    pass


def _column_slot(context, code, row_reader_type):
    """Recognize exactly scope['table']['column'] for a single-table context."""
    if len(context.tables) != 1 or type(code) is not types.CodeType:
        return None
    try:
        tree = ast.parse(code.co_filename, mode='eval')
        outer = tree.body
        if type(outer) is not ast.Subscript or type(outer.value) is not ast.Subscript:
            return None
        inner = outer.value
        if type(inner.value) is not ast.Name or inner.value.id != 'scope':
            return None
        if type(inner.slice) is not ast.Constant or type(outer.slice) is not ast.Constant:
            return None
        if type(inner.slice.value) is not str or type(outer.slice.value) is not str:
            return None
        if code != compile(tree, code.co_filename, 'eval', dont_inherit=True, optimize=2):
            return None
        reader = context.row_readers.get(inner.slice.value)
        if type(reader) is row_reader_type and reader is context.table.reader:
            return reader.columns.get(outer.slice.value)
    except (SyntaxError, TypeError, ValueError, RecursionError):
        return None
    return None


def install():
    """Install scoped wrappers and return a function restoring the original methods."""
    from sqlglot.executor.context import Context
    from sqlglot.executor.python import PythonExecutor
    from sqlglot.executor.table import RowReader, Table

    generate_tuple = PythonExecutor.generate_tuple
    eval_tuple = Context.eval_tuple
    hash_join = PythonExecutor.hash_join
    programs = {}

    def generate(self, expressions):
        codes = generate_tuple(self, expressions)
        if type(codes) is not tuple or not 2 <= len(codes) <= 64:
            return codes
        result = _Projection(codes)
        result.program = None
        result.warmup = 12
        return result

    def evaluate(self, codes):
        if type(self) is Context and type(codes) is _Projection:
            if codes.program is not None:
                return eval(codes.program, self.env)
            if codes.warmup:
                codes.warmup -= 1
                if not codes.warmup and all(type(code) is types.CodeType for code in codes):
                    key = tuple(codes)
                    if key not in programs and len(programs) < 4096:
                        programs[key] = _fuse(key)
                    codes.program = programs.get(key)
        return eval_tuple(self, codes)

    def key_rows(context, codes):
        if len(codes) == 1:
            code = codes[0]
            column = _column_slot(context, code, RowReader)
            if column is not None:
                for reader, _ in context:
                    row = reader.row
                    value = row[column]
                    if value is not None:
                        yield (value,), row
                return
            for reader, ctx in context:
                value = ctx.eval(code)
                if value is not None:
                    yield (value,), reader.row
            return
        for reader, ctx in context:
            key = ctx.eval_tuple(codes)
            if all(value is not None for value in key):
                yield key, reader.row

    def inner_join(self, join, source_context, join_context, condition, condition_context):
        if (type(self) is not PythonExecutor or type(source_context) is not Context
                or type(join_context) is not Context or join.get('side') not in ('', None)):
            return hash_join(self, join, source_context, join_context, condition, condition_context)

        # An inner join emits matches only. Preserve key insertion order from the
        # source and row order on both sides. Right-only keys produce no rows.
        source_codes = self.generate_tuple(join['source_key'])
        join_codes = self.generate_tuple(join['join_key'])
        groups = {}
        for key, row in key_rows(source_context, source_codes):
            group = groups.get(key)
            if group is None:
                group = groups[key] = ([], [])
            group[0].append(row)
        for key, row in key_rows(join_context, join_codes):
            group = groups.get(key)
            if group is not None:
                group[1].append(row)

        table = Table(source_context.columns + join_context.columns)
        if condition:
            matches = self._join_matches
            for left, right in groups.values():
                for source_row, join_row in product(left, right):
                    row = source_row + join_row
                    if matches(row, condition, condition_context):
                        table.append(row)
        else:
            append = table.append
            for left, right in groups.values():
                for source_row, join_row in product(left, right):
                    append(source_row + join_row)
        return table

    PythonExecutor.generate_tuple = generate
    PythonExecutor.hash_join = inner_join
    Context.eval_tuple = evaluate

    def restore():
        Context.eval_tuple = eval_tuple
        PythonExecutor.hash_join = hash_join
        PythonExecutor.generate_tuple = generate_tuple

    return restore
