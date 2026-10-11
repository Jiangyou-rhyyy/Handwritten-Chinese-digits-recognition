import unittest

from app import interpolate_points


class DrawingHelpersTests(unittest.TestCase):
    def test_interpolation_keeps_fast_pointer_strokes_continuous(self):
        points = interpolate_points((0, 0), (20, 0), max_step=5)

        self.assertEqual(points[0], (0, 0))
        self.assertEqual(points[-1], (20, 0))
        self.assertTrue(
            all(abs(right[0] - left[0]) <= 5 for left, right in zip(points, points[1:]))
        )


if __name__ == "__main__":
    unittest.main()
