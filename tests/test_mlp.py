import unittest
import numpy as np
import os, sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..')
sys.path.append( mymodule_dir )

from data.dataset import Dataset
from models.mlp import MLP
import random

np.random.seed(10)
random.seed(10)

class TestMLP(unittest.TestCase):
    def test_mlp_set_weights(self):
        ds= Dataset("datasets/xnor.data")
        nn = MLP(ds, 2)
        w1 = np.array([[-30,20,20],[10,-20,-20]])
        w2 = np.array([[-10,20,20]])
        nn.setWeights(w1, w2)
        self.assertAlmostEqual(nn.predict(np.array([0,0]) )[0], 0.99995456)
        self.assertAlmostEqual( nn.predict(np.array([0,1]) )[0], 4.54803785e-05)
        self.assertAlmostEqual( nn.predict(np.array([1,0]) )[0], 4.54803785e-05)
        self.assertAlmostEqual( nn.predict(np.array([1,1]) )[0], 0.99995456)
        self.assertEqual(nn.costFunction(),  0.9999090846497954)

    def test_mlp_random_weights(self):
        ds= Dataset("datasets/xnor.data")
        nn = MLP(ds, 5)
        nn.build_model()
        self.assertAlmostEqual( nn.predict(np.array([0,0]) )[0], 0.50023085100395)
        self.assertAlmostEqual( nn.predict(np.array([0,1]) )[0], 0.4997757337395438)
        self.assertAlmostEqual( nn.predict(np.array([1,0]) )[0], 0.4996371044295929)
        self.assertAlmostEqual( nn.predict(np.array([1,1]) )[0], 0.5003572439153007)
        self.assertEqual(nn.costFunction(), 0.5000001814519759)

    def test_mlp_random_weights_and_normalization(self):
        ds= Dataset("datasets/xnor.data")
        nn = MLP(ds, 3, normalize=True)
        nn.build_model()
        self.assertAlmostEqual( nn.predict(np.array([0,0]) )[0], 0.5003803053172108)
        self.assertAlmostEqual( nn.predict(np.array([0,1]) )[0], 0.4995336454467019)
        self.assertAlmostEqual( nn.predict(np.array([1,0]) )[0], 0.499631214661103)
        self.assertAlmostEqual( nn.predict(np.array([1,1]) )[0], 0.5004465291720784)
        self.assertEqual(nn.costFunction(), 0.5000003487548157)

def run():
    unittest.main()

if __name__ == '__main__':
    run()