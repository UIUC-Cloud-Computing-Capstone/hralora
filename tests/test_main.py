"""
Unit tests for main.py

Tests cover: merge_config (copying args onto config). The __main__ block
(config load, Accelerator, training loop) is not unit-tested here.
"""
import argparse
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import merge_config


# -----------------------------------------------------------------------------
# merge_config
# -----------------------------------------------------------------------------
class TestMergeConfig(unittest.TestCase):
    def test_copies_all_args_onto_config(self):
        config = argparse.Namespace()
        args = argparse.Namespace()
        args.a = 1
        args.b = "two"
        args.seed = 42
        out = merge_config(config, args)
        self.assertIs(out, config)
        self.assertEqual(config.a, 1)
        self.assertEqual(config.b, "two")
        self.assertEqual(config.seed, 42)

    def test_overwrites_existing_config_attrs(self):
        config = argparse.Namespace()
        config.x = 0
        args = argparse.Namespace()
        args.x = 10
        merge_config(config, args)
        self.assertEqual(config.x, 10)

    def test_empty_args_leaves_config_unchanged(self):
        config = argparse.Namespace()
        config.only = "here"
        args = argparse.Namespace()
        out = merge_config(config, args)
        self.assertIs(out, config)
        self.assertEqual(config.only, "here")


if __name__ == "__main__":
    unittest.main()
