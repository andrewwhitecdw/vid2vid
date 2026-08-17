'''Regression tests for wrapped flowNet assignment in training pipeline.'''
import ast
import os


def _collect_wrap_model_assignments(func_name):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, 'models', 'models.py')
    with open(path) as f:
        source = f.read()

    tree = ast.parse(source)
    func = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == func_name
    )

    assignments = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        value = node.value
        is_wrap_model_call = (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == 'wrap_model'
        )
        if is_wrap_model_call:
            names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
            assignments.append(names)

    return assignments


def test_create_model_assigns_wrapped_flow_net():
    assignments = _collect_wrap_model_assignments('create_model')
    assert any(names == ['modelG', 'modelD', 'flowNet'] for names in assignments), (
        f'Expected create_model to assign wrap_model result to flowNet, got {assignments}'
    )
    assert all('flownet' not in names for names in assignments)


def test_create_optimizer_assigns_wrapped_flow_net():
    assignments = _collect_wrap_model_assignments('create_optimizer')
    assert any(names == ['modelG', 'modelD', 'flowNet'] for names in assignments), (
        f'Expected create_optimizer to assign wrap_model result to flowNet, got {assignments}'
    )
    assert all('flownet' not in names for names in assignments)
