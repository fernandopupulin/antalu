"""Coverage certificates for an unchanged prefix of the pinned TPC-DS tasks."""

import hashlib
import marshal
import os
import zlib

import stages


TEST_ID = 'tests/test_executor.py::TestExecutor::test_execute_tpcds'
VERSION = 1
PREFIX = 'sqlglot/optimizer/'


def code_key(root, code):
    path = code.co_filename
    prefix = root + '/' + PREFIX
    if path.startswith(prefix):
        return path[len(root) + 1:], code.co_qualname, code.co_firstlineno
    return None


def optimizer_state(alias=None):
    import sys
    import statefp

    modules = {name for name in sys.modules if name.startswith('sqlglot.optimizer')}
    state, _ = statefp.snapshot(alias, only=modules)
    return hashlib.blake2b(marshal.dumps(tuple(sorted(state.items())), 2), digest_size=16).digest()


def load(path):
    try:
        with open(path, 'rb') as file:
            version, probe, shared, queries = marshal.loads(zlib.decompress(file.read()))
        if version == VERSION and probe == stages.hash_probe() and type(queries) is dict:
            return frozenset(shared), queries
    except (OSError, EOFError, TypeError, ValueError, zlib.error):
        pass
    return None


def reusable_prefix(path, root, pairs, active, late):
    """Stop reusing as soon as any task may reach a changed function or alter state."""
    cached = load(path)
    if cached is None or not pairs:
        return {}, {}, None
    changed = {code_key(root, old) for old, _ in pairs}
    shared, queries = cached
    if None in changed or changed & shared:
        return {}, {}, None
    prefix = {}
    anchors = {}
    initial = None
    for i, (inputs, state, functions, quiet) in sorted(queries.items()):
        if inputs is None or i in late or changed.intersection(functions):
            break
        if not quiet:
            if prefix or (active is not None and i not in active):
                break
            anchors[i] = inputs
            continue
        if active is None or i in active:
            if not prefix:
                initial = state
            prefix[i] = inputs
    return prefix, anchors if prefix else {}, initial
