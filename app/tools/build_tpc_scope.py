"""Record the pinned test's optimizer calls and per-task state with the real executor."""

import marshal
import os
import sys
import zlib

import build_stages
import runner
import stages
import statefp
import tpc_scope
from reduce import STATE


def record_once(root):
    test_id = tpc_scope.TEST_ID
    build_stages.parent_setup(root, [{'file': test_id.split('::')[0], 'id': test_id}])
    case = runner.load_case(test_id, 0)
    test_class = type(case)
    module = sys.modules[test_class.__module__]
    original_helper = test_class.subtestHelper
    sqls = test_class.tpcds_sqls
    assert len({sql for _, sql, _ in sqls}) == len(sqls)
    queries, shared = {}, set()
    active = None
    rows = {}
    monitor = sys.monitoring
    tool = 3

    def called(code, _offset):
        key = tpc_scope.code_key(root, code)
        if key is None:
            return monitor.DISABLE
        (shared if active is None else active).add(key)

    def execute(index, fn, args, kwargs):
        nonlocal active
        if not args[1].get('execute'):
            return fn(*args, **kwargs)
        before, _ = statefp.snapshot()
        inputs = stages.fingerprint((args, kwargs, module._schema))
        state = tpc_scope.optimizer_state()
        functions = set()
        active = functions
        try:
            value = fn(*args, **kwargs)
        finally:
            active = None
        after, _ = statefp.snapshot()
        queries[index] = [inputs, state, functions, inputs is not None and before == after]
        return value

    def compare(case, index, table, tpch=True):
        nonlocal active
        rows[index] = stages.rows_fingerprint(table)
        before, _ = statefp.snapshot()
        active = queries[index][2]
        try:
            return original_helper(case, index, table, tpch=tpch)
        finally:
            active = None
            after, _ = statefp.snapshot()
            queries[index][3] &= before == after

    monitor.use_tool_id(tool, 'tpc-query-dependencies')
    monitor.register_callback(tool, monitor.events.PY_START, called)
    monitor.register_callback(tool, monitor.events.PY_RESUME, called)
    monitor.set_events(tool, monitor.events.PY_START | monitor.events.PY_RESUME)
    STATE.task_hook = execute
    test_class.subtestHelper = compare
    try:
        passed = build_stages.run_one(case, 0, test_id, None)
    finally:
        STATE.task_hook = None
        test_class.subtestHelper = original_helper
        monitor.set_events(tool, 0)
        monitor.free_tool_id(tool)

    expected = stages.load_map('/app/maps/stages.bin')[test_id]['tasks']
    assert passed and rows == {i: data[3] for i, data in expected.items()}
    assert queries.keys() == expected.keys()
    return shared, queries


if __name__ == '__main__':
    root, output = sys.argv[1:3]
    shared, queries = record_once(os.path.abspath(root))
    compact = {i: (a, b, tuple(sorted(c)), d) for i, (a, b, c, d) in sorted(queries.items())}
    data = (tpc_scope.VERSION, stages.hash_probe(), tuple(sorted(shared)), compact)
    with open(output, 'wb') as file:
        file.write(zlib.compress(marshal.dumps(data, 2), 9))
    print(len(queries), 'queries;', sum(row[3] for row in queries.values()), 'state-neutral')
