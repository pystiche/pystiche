import unittest
from math import factorial
from pystiche.misc import misc


class Tester(unittest.TestCase):
    def test_prod(self):
        n = 10
        iterable = range(1, n + 1)

        actual = misc.prod(iterable)
        desired = factorial(n)
        self.assertEqual(actual, desired)

    def test_to_1d_arg(self):
        val = 0
        actual = misc.to_1d_arg(val)
        desired = (val,)
        self.assertTupleEqual(actual, desired)

        val = (0,)
        actual = misc.to_1d_arg(val)
        desired = val
        self.assertTupleEqual(actual, desired)

        val = 0
        actual = misc.to_1d_arg([val])
        desired = (val,)
        self.assertTupleEqual(actual, desired)

        val = (0, 0)
        with self.assertRaises(AssertionError):
            misc.to_1d_arg(val)

    def test_to_2d_arg(self):
        val = 0
        actual = misc.to_2d_arg(val)
        desired = (val, val)
        self.assertTupleEqual(actual, desired)

        val = (0, 0)
        actual = misc.to_2d_arg(val)
        desired = val
        self.assertTupleEqual(actual, desired)

        val = 0
        actual = misc.to_2d_arg([val] * 2)
        desired = (val, val)
        self.assertTupleEqual(actual, desired)

        val = (0,)
        with self.assertRaises(AssertionError):
            misc.to_2d_arg(val)

    def test_to_3d_arg(self):
        val = 0
        actual = misc.to_3d_arg(val)
        desired = (val, val, val)
        self.assertTupleEqual(actual, desired)

        val = (0, 0, 0)
        actual = misc.to_3d_arg(val)
        desired = val
        self.assertTupleEqual(actual, desired)

        val = 0
        actual = misc.to_3d_arg([val] * 3)
        desired = (val, val, val)
        self.assertTupleEqual(actual, desired)

        val = (0,)
        with self.assertRaises(AssertionError):
            misc.to_3d_arg(val)

    def test_zip_equal(self):
        foo = (1, 2)
        bar = ("a", "b")

        actual = tuple(misc.zip_equal(foo, bar))
        desired = tuple(zip(foo, bar))
        self.assertTupleEqual(actual, desired)

        foo = (1, 2)
        bar = ("a", "b", "c")

        with self.assertRaises(RuntimeError):
            misc.zip_equal(foo, bar)

    def test_to_eng(self):
        def assertEqual(actual, desired):
            self.assertAlmostEqual(actual[0], desired[0])
            self.assertEqual(actual[1], desired[1])

        val = 123456789
        actual = misc.to_eng(val)
        desired = (123.456789, 6)
        assertEqual(actual, desired)

        val = -9876e-5
        actual = misc.to_eng(val)
        desired = (-98.76, -3)
        assertEqual(actual, desired)

    def test_to_eng_zero(self):
        eps = 1e-8
        vals = (0.0, 1e-1 * eps, -1e-1 * eps)

        for val in vals:
            sig, exp = misc.to_eng(val, eps=eps)
            self.assertAlmostEqual(sig, 0.0)
            self.assertEqual(exp, 0)

    def test_to_engstr(self):
        val = 123456789
        actual = misc.to_engstr(val)
        desired = "123.5e6"
        self.assertEqual(actual, desired)

        val = -98765e-8
        actual = misc.to_engstr(val)
        desired = "-987.7e-6"
        self.assertEqual(actual, desired)

    def test_to_engstr_zero(self):
        eps = 1e-8
        vals = (0.0, 1e-1 * eps, -1e-1 * eps)

        for val in vals:
            self.assertEqual(misc.to_engstr(val, eps=eps), "0")

    def test_to_engstr_smaller_than_kilo(self):
        val = 123.456
        actual = misc.to_engstr(val)
        desired = "123.5"
        self.assertEqual(actual, desired)

    def test_to_engstr_larger_than_milli(self):
        val = 123.456e-3
        actual = misc.to_engstr(val)
        desired = "0.1235"
        self.assertEqual(actual, desired)

    @unittest.skipIf(True, "FIXME")
    def test_to_engstr_digits(self):
        val = 123456

        digits = 2
        actual = misc.to_engstr(val, digits=digits)
        desired = "12e3"
        self.assertEqual(actual, desired)

        digits = 5
        actual = misc.to_engstr(val, digits=digits)
        desired = "123.46e3"
        self.assertEqual(actual, desired)

    @unittest.skipIf(True, "FIXME")
    def test_to_tuplestr(self):
        seq = ()
        actual = misc.to_tuplestr(seq)
        desired = str(seq)
        self.assertEqual(actual, desired)

        seq = (0.0,)
        actual = misc.to_tuplestr(seq)
        desired = str(seq)
        self.assertEqual(actual, desired)

        seq = (0.0, 1.0)
        actual = misc.to_tuplestr(seq)
        desired = str(seq)
        self.assertEqual(actual, desired)

        seq = (0.0, "a")
        actual = misc.to_tuplestr(seq)
        desired = str(seq)
        self.assertEqual(actual, desired)
