import unittest
from drill_ir.analysis.cardinality_confirmatory import apply_confirmatory_case,evaluate_confirmatory_case


class ConfirmatoryTest(unittest.TestCase):
    def fixture(self):
        gold={}; rankings={}
        for qid in range(1,21):
            first=qid*10
            if qid%2: gold[qid]=(first,); scores=[.95,.2,.1,.05,.01]
            else: gold[qid]=(first,first+1); scores=[.95,.9,.1,.05,.01]
            rankings[qid]=([first+i for i in range(5)],scores)
        return gold,rankings

    def test_all_cases(self):
        gold,rankings=self.fixture()
        for case in ('baseline_multiclass','cost_sensitive_multiclass','multiclass_ridge_ensemble'):
            metrics,deployment,fitted=evaluate_confirmatory_case(case,rankings,gold,outer_folds=5,inner_folds=5,seed=42,maximum_k=5,C=.1,alpha=1,lambda_values=[.25,.5,.75])
            self.assertEqual(metrics['f2'],1.0)
            selected,_=apply_confirmatory_case(case,fitted,rankings,5,deployment)
            self.assertEqual(selected,{qid:list(aids) for qid,aids in gold.items()})


if __name__=='__main__': unittest.main()
