__author__ = 'Charlie'


import unittest
import data_manager
import numpy as np


class TestDataManager(unittest.TestCase):

    def test_generate_rotations(self):
        imgs = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ], dtype=np.float32)

        clss = np.array([
            1, 2
        ], dtype=np.int32)

        rotated_imgs, rotated_clss = data_manager.generate_rotated_images(imgs, clss)

        expected_rotated_imgs = np.array([
            [[1, 2], [3, 4]],
            [[2, 4], [1, 3]],
            [[4, 3], [2, 1]],
            [[3, 1], [4, 2]],
            [[5, 6], [7, 8]],
            [[6, 8], [5, 7]],
            [[8, 7], [6, 5]],
            [[7, 5], [8, 6]]
        ], dtype=np.float32)

        expected_rotated_clss = np.array([
            1, 1, 1, 1, 2, 2, 2, 2
        ], dtype=np.int32)

        self.assertTrue(np.array_equal(rotated_imgs, expected_rotated_imgs))
        self.assertTrue(np.array_equal(rotated_clss, expected_rotated_clss))