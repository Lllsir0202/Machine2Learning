import unittest
import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from utils.activations.methods.softmax import Softmax

class TestSoftmax(unittest.TestCase):
    """为 Softmax 激活函数编写的单元测试"""

    def test_forward(self):
        """测试 forward 方法"""
        input_array = np.array([0, 1, 2])
        exp_x = np.exp(input_array)
        expected_output = exp_x / np.sum(exp_x)
        
        # 测试计算结果
        np.testing.assert_allclose(Softmax.forward(input_array), expected_output)
        
        # 测试所有输出的总和是否为1
        self.assertAlmostEqual(np.sum(Softmax.forward(input_array)), 1.0)

    def test_forward_stability(self):
        """测试数值稳定性"""
        # 一个包含较大数值的输入
        input_array = np.array([1000, 1001, 1002])
        # 减去最大值以保证数值稳定
        stable_input = input_array - np.max(input_array)
        exp_x = np.exp(stable_input)
        expected_output = exp_x / np.sum(exp_x)
        
        np.testing.assert_allclose(Softmax.forward(input_array), expected_output)
        self.assertAlmostEqual(np.sum(Softmax.forward(input_array)), 1.0)

if __name__ == '__main__':
    unittest.main()