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


class FlatMatchNCSAMWarmupTest(unittest.TestCase):
    def test_flatmatch_train_uses_shared_flatmatch_step(self):
        self.assertTrue(hasattr(trainer.FreeMatchTrainer, '_train_flatmatch_batch'))

        tree = _method_tree(trainer.FreeMatchTrainer, 'train')
        self.assertTrue(_calls_method(tree, '_train_flatmatch_batch'))

    def test_flatmatch_amp_unscales_before_merging_base_grad(self):
        tree = _method_tree(trainer.FreeMatchTrainer, '_train_flatmatch_batch')
        self.assertTrue(_calls_method(tree, 'unscale_'))
        self.assertTrue(_calls_method(tree, '_restore_perturbation_and_add_base_grads'))

    def test_ncsam_plain_phase_uses_shared_flatmatch_step(self):
        self.assertTrue(hasattr(flatmatch_ncsam_trainer.FlatMatchNCSAMTrainer, '__use_plain_flatmatch_step__'))

        tree = _method_tree(flatmatch_ncsam_trainer.FlatMatchNCSAMTrainer, 'train')
        self.assertTrue(_calls_method(tree, '__use_plain_flatmatch_step__'))
        self.assertTrue(_calls_method(tree, '_train_flatmatch_batch'))


if __name__ == '__main__':
    unittest.main()
