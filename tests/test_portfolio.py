import unittest
import numpy as np
import pandas as pd
from src.quickstart import load_survey, benchmark, allocate, ROOT
from src.monitor import monitor
class PortfolioTests(unittest.TestCase):
    def test_data_contract(self):
        d,x,y=load_survey()
        self.assertEqual(x.shape,(400,30))
        self.assertEqual(int((y>=7).sum()),120)
    def test_budget(self):
        r=allocate([.9,.7,.2,.1],[5,4,3,2],9,2)
        self.assertEqual(r["contacts"],2)
        self.assertLessEqual(r["cost"],9)
        self.assertAlmostEqual(r["expected_interest"],1.6)
    def test_monitor(self):
        d=pd.read_csv(ROOT/"results/monitoring_stress.csv")
        self.assertEqual(monitor(d.iloc[0],d.iloc[0])["status"],"PASS")
        self.assertEqual(monitor(d.iloc[0],d.iloc[1])["status"],"REVIEW")
    def test_archived_results(self):
        d=pd.read_csv(ROOT/"results/model_benchmark.csv")
        self.assertLess(float(d[d.model=="Factor OLS"].RMSE.iloc[0]),float(d[d.model=="OLS"].RMSE.iloc[0]))
        r=pd.read_csv(ROOT/"results/root_cause.csv")
        self.assertEqual(int(r.fn.sum()),58)
        self.assertEqual(int(r.tp.sum()),62)
if __name__=="__main__":
    unittest.main()
