import unittest
import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from utils.activations.methods.leakyrelu import Leaky_Relu

class TestLeakyRelu(unittest.TestCase):
    """为 Leaky ReLU 激活函数编写的单元测试"""

    def test_forward(self):
        """测试 forward 方法"""
        alpha = 0.01
        input_array = np.array([-2, -0.5, 0, 0.5, 2])
        expected_output = np.array([-2 * alpha, -0.5 * alpha, 0, 0.5, 2])
        np.testing.assert_allclose(Leaky_Relu.forward(input_array, alpha), expected_output)

    def test_backward(self):
        """测试 backward 方法"""
        alpha = 0.01
        input_array = np.array([-2, -0.5, 0, 0.5, 2])
        # Leaky ReLU的导数在x>=0时为1，在x<0时为alpha
        expected_output = np.array([alpha, alpha, 1, 1, 1])
        np.testing.assert_allclose(Leaky_Relu.backward(input_array, alpha), expected_output)

if __name__ == '__main__':
    unittest.main()