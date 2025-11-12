import unittest
import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from utils.activations.methods.relu import Relu

class TestRelu(unittest.TestCase):
    """为 ReLU 激活函数编写的单元测试"""

    def test_forward(self):
        """测试 forward 方法"""
        input_array = np.array([-2, -0.5, 0, 0.5, 2])
        expected_output = np.array([0, 0, 0, 0.5, 2])
        np.testing.assert_allclose(Relu.forward(input_array), expected_output)

    def test_backward(self):
        """测试 backward 方法"""
        input_array = np.array([-2, -0.5, 0, 0.5, 2])
        # ReLU的导数在x>0时为1，在x<=0时为0
        expected_output = np.array([0, 0, 0, 1, 1])
        np.testing.assert_allclose(Relu.backward(input_array), expected_output)

if __name__ == '__main__':
    unittest.main()