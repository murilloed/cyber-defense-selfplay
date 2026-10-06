import unittest,torch
from cyberplay.core import reward,train
class Tests(unittest.TestCase):
    def test_blocking_and_zero_sum(self):
        a=torch.tensor([0,1,2]); self.assertTrue(torch.equal(reward(a,a),torch.zeros(3)))
        r=reward(a,torch.tensor([1,2,0])); self.assertEqual(r.tolist(),[1,2,3])
        self.assertTrue(torch.equal(r+(-r),torch.zeros(3)))
    def test_finite_training(self):
        a,d,_,h=train(2,1)
        self.assertEqual(len(h),2)
        self.assertTrue(all(bool(torch.isfinite(p).all()) for p in d.parameters()))
