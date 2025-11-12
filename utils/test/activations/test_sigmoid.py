import unittest
import numpy as np
import sys
import os

# 为了让测试脚本能够找到需要测试的模块，我们需要将项目根目录添加到 Python 路径中
# 这是一种常见的做法，可以确保无论从哪里运行测试，导入都能正常工作
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from utils.activations.methods.sigmoid import Sigmoid

class TestSigmoid(unittest.TestCase):
    """
    为 Sigmoid 激活函数编写的单元测试
    """

    def test_forward_single_value(self):
        """测试 forward 方法处理单个数值"""
        # sigmoid(0) 应该等于 0.5
        self.assertAlmostEqual(Sigmoid.forward(0), 0.5)
        # 测试一个正数
        self.assertAlmostEqual(Sigmoid.forward(1), 1 / (1 + np.exp(-1)))
        # 测试一个负数
        self.assertAlmostEqual(Sigmoid.forward(-1), 1 / (1 + np.exp(1)))

    def test_forward_numpy_array(self):
        """测试 forward 方法处理 numpy 数组"""
        input_array = np.array([-1, 0, 1])
        expected_output = np.array([1 / (1 + np.exp(1)), 0.5, 1 / (1 + np.exp(-1))])
        # 使用 np.testing.assert_allclose 来比较浮点数数组，它能处理微小的精度差异
        np.testing.assert_allclose(Sigmoid.forward(input_array), expected_output)

    def test_backward_single_value(self):
        """测试 backward 方法处理单个数值"""
        # sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        self.assertAlmostEqual(Sigmoid.backward(0), 0.25)
        
        s_1 = 1 / (1 + np.exp(-1))
        self.assertAlmostEqual(Sigmoid.backward(1), s_1 * (1 - s_1))

    def test_backward_numpy_array(self):
        """测试 backward 方法处理 numpy 数组"""
        input_array = np.array([-1, 0, 1])
        s = 1 / (1 + np.exp(-input_array))
        expected_output = s * (1 - s)
        np.testing.assert_allclose(Sigmoid.backward(input_array), expected_output)

# 这使得脚本可以直接运行
if __name__ == '__main__':
    unittest.main()