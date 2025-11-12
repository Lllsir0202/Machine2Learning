import unittest
import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from utils.activations.methods.tanh import Tanh

class TestTanh(unittest.TestCase):
    """为 Tanh 激活函数编写的单元测试"""

    def test_forward(self):
        """测试 forward 方法"""
        input_array = np.array([-1, 0, 1])
        expected_output = np.tanh(input_array)
        np.testing.assert_allclose(Tanh.forward(input_array), expected_output)
        self.assertAlmostEqual(Tanh.forward(0), 0)

    def test_backward(self):
        """测试 backward 方法"""
        input_array = np.array([-1, 0, 1])
        # Tanh的导数是 1 - tanh(x)^2
        expected_output = 1 - np.tanh(input_array)**2
        np.testing.assert_allclose(Tanh.backward(input_array), expected_output)
        self.assertAlmostEqual(Tanh.backward(0), 1)

if __name__ == '__main__':
    unittest.main()