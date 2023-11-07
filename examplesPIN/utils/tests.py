import doctest

from utils import load_ur5_parallel


def load_tests(loader, tests, pattern):
    tests.addTests(doctest.DocTestSuite(load_ur5_parallel))
    return tests
