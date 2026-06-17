import ast
import inspect
import textwrap
import unittest

import flatmatch_ncsam_trainer
import trainer


def _method_tree(cls, method_name):
    source = textwrap.dedent(inspect.getsource(getattr(cls, method_name)))
    return ast.parse(source)


def _calls_method(tree, method_name):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == method_name:
            return True
    return False


def _method_call_count(tree, method_name):
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == method_name:
            count += 1
    return count


class FlatMatchNCSAMWarmupTest(unittest.TestCase):
    def test_flatmatch_train_uses_shared_flatmatch_step(self):
        self.assertTrue(hasattr(trainer.FreeMatchTrainer, '_train_flatmatch_batch'))

        tree = _method_tree(trainer.FreeMatchTrainer, 'train')
        self.assertTrue(_calls_method(tree, '_train_flatmatch_batch'))

    def test_flatmatch_step_matches_original_amp_xsharp_behavior(self):
        tree = _method_tree(trainer.FreeMatchTrainer, '_train_flatmatch_batch')
        self.assertFalse(_calls_method(tree, 'unscale_'))
        self.assertFalse(_calls_method(tree, '_restore_perturbation_and_add_base_grads'))

    def test_ncsam_plain_phase_uses_shared_flatmatch_step(self):
        self.assertTrue(hasattr(flatmatch_ncsam_trainer.FlatMatchNCSAMTrainer, '__use_plain_flatmatch_step__'))

        tree = _method_tree(flatmatch_ncsam_trainer.FlatMatchNCSAMTrainer, 'train')
        self.assertTrue(_calls_method(tree, '__use_plain_flatmatch_step__'))
        self.assertTrue(_calls_method(tree, '_train_flatmatch_batch'))
        self.assertEqual(_method_call_count(tree, 'unscale_'), 1)


if __name__ == '__main__':
    unittest.main()
