import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.track_a import ood_threshold_analysis as oa
from src.track_a import ood_threshold_experiment as oe


def validation(scores):
    return pd.DataFrame(dict(partition=['validation'] * len(scores), max_probability=scores))


def sample():
    known = ['a', 'benign']
    labels = pd.Series(['a', 'benign', 'a', 'benign', 'a', 'benign', 'new', 'new', 'other'])
    parts = pd.Series(['train', 'train', 'validation', 'validation', 'test', 'test', 'unknown', 'unknown', 'unknown'])
    ids = [2, 3, 4, 5, 6, 7, 8]
    probs = np.array([[.8,.2],[.4,.6],[.7,.3],[.3,.7],[.55,.45],[.95,.05],[.6,.4]])
    p = pd.DataFrame(dict(row_id=ids, partition=parts.loc[ids].to_numpy(), actual=labels.loc[ids].to_numpy(),
                          predicted=np.array(known)[probs.argmax(1)], max_probability=probs.max(1),
                          prob_00=probs[:,0], prob_01=probs[:,1]))
    assignments = pd.DataFrame(dict(row_id=labels.index, partition=parts))
    patterns = pd.Series(['x','y','v','w','t','z','shared','shared','single'])
    return known, labels, parts, p, assignments, patterns


class ThresholdTests(unittest.TestCase):
    def test_ties_do_not_get_split_to_fill_budget(self):
        p = validation([.2] * 10 + [.8] * 90)
        q = oa.select_threshold(p, 5)
        self.assertEqual((q['threshold'], q['validation_reject_rows'], q['allowed_reject_rows']), (.2, 0, 5))
        self.assertEqual(q['boundary_tie_rows'], 10)
        self.assertEqual(oa.select_threshold(p, 10)['validation_reject_rows'], 10)

    def test_strict_boundary_all_equal_and_integer_budget(self):
        q = oa.select_threshold(validation([.75] * 101), 5)
        self.assertEqual((q['threshold'], q['validation_reject_rows'], q['allowed_reject_rows']), (.75, 0, 5))
        q = oa.select_threshold(validation(np.arange(1, 101)/100), 5)
        self.assertEqual((q['threshold'], q['validation_reject_rows']), (.06, 5))
        self.assertFalse(.06 < q['threshold'])
        self.assertTrue(np.nextafter(.06, 0) < q['threshold'])

    def test_only_validation_and_valid_scores_are_accepted(self):
        for p in [validation([]), validation([np.nan]), validation([1.01]), validation([-.01]),
                  validation([.8]).assign(partition='unknown')]:
            with self.assertRaises(ValueError): oa.select_threshold(p, 5)

    def test_changing_test_or_unknown_never_changes_thresholds(self):
        *_, p, assignments, patterns = sample()
        before = oa.policies_for(p)[0]
        changed = p.copy()
        changed.loc[changed.partition.ne('validation'), 'max_probability'] = 0.
        self.assertEqual(before, oa.policies_for(changed)[0])


class ContractTests(unittest.TestCase):
    def test_probability_round_trip_and_alignment(self):
        known, y, part, p, a, _ = sample()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'p.csv'; p.to_csv(path,index=False)
            read = pd.read_csv(path, float_precision='round_trip')
            got = oa.validate_saved(read.sample(frac=1,random_state=1), a, y, part, known)
            np.testing.assert_array_equal(got.prob_00, p.prob_00)
        for column, value in [('prob_00', .99), ('max_probability', .12), ('predicted', 'new'), ('actual', 'new')]:
            bad=p.copy();bad.loc[0,column]=value
            with self.assertRaises(ValueError):oa.validate_saved(bad,a,y,part,known)
        with self.assertRaises(ValueError):oa.validate_saved(pd.concat([p,p.iloc[:1]]),a,y,part,known)
        with self.assertRaises(ValueError):oa.validate_saved(p.iloc[1:],a,y,part,known)

    def test_max_argmax_tie_and_bad_class_order(self):
        known,y,part,p,a,_=sample();p.loc[0,['prob_00','prob_01','max_probability','predicted']]=[.5,.5,.5,'a']
        oa.validate_saved(p,a,y,part,known)
        with self.assertRaises(ValueError):oa.validate_saved(p,a,y,part,known[::-1])


class EvaluationTests(unittest.TestCase):
    def test_pattern_equal_weighting_is_not_any_or_row_weighting(self):
        known,y,part,p,a,patterns=sample()
        policy=dict(policy_id='test',target_reject_rate=.05,threshold=.6)
        r,cm,classes,patterns_table,summary=oa.evaluate_policy(p,known,patterns,policy,{})
        # Only one of two rows in the 'new' pattern is rejected; score==.6 is accepted.
        self.assertEqual(r['unknown_reject_rows'],1)
        self.assertAlmostEqual(r['unknown_reject_rate'],1/3)
        self.assertAlmostEqual(r['unknown_class_macro_reject_rate'],.25)
        self.assertAlmostEqual(r['unknown_pattern_macro_reject_rate'],.25)
        self.assertEqual(len(cm),9)
        self.assertEqual(cm['count'].sum(),5)
        self.assertEqual(summary.set_index('label').loc['new','pattern_equal_weight_reject_rate'],.5)
        none=oa.evaluate_policy(p,known,patterns,dict(policy_id='none',target_reject_rate=np.nan,threshold=0),{})[0]
        self.assertEqual((none['known_reject_rows'],none['unknown_reject_rows']),(0,0))

    def test_zero_accepted_known_is_na_and_normal_outcomes_are_distinct(self):
        known,y,part,p,a,patterns=sample()
        r,*_=oa.evaluate_policy(p,known,patterns,dict(policy_id='all',target_reject_rate=.1,threshold=1.),{})
        self.assertEqual(r['accepted_known_rows'],0)
        self.assertTrue(np.isnan(r['accepted_known_accuracy']))
        self.assertEqual((r['benign_reject_rows'],r['benign_wrong_known_rows']),(1,0))
        self.assertAlmostEqual(oa.auc_report(p,{})['auroc'],2/3)

    def test_train_only_pattern_dictionary_and_class_denominators(self):
        known,y,part,p,a,patterns=sample()
        audit=oa.partition_audit(y,part,patterns,{})
        row=audit[(audit.partition=='unknown')&(audit.label=='__all__')].iloc[0]
        self.assertEqual((row.rows,row.unique_patterns,row.train_pattern_seen_rows),(3,2,0))
        patterns.iloc[0]='shared'
        row=oa.partition_audit(y,part,patterns,{}).query("partition == 'unknown' and label == '__all__'").iloc[0]
        self.assertEqual(row.train_pattern_seen_rows,2)


class DriverTests(unittest.TestCase):
    def test_failed_training_leaves_incomplete_manifest(self):
        known,y,part,p,a,patterns=sample()
        frame=pd.DataFrame({'Label':y, **{name:[1]*len(y) for name in oe.feature_groups()['all']}})
        outer=part.replace('validation','train')
        splits=pd.DataFrame({f'{s}_{seed}':outer for s in ['random','pattern_group'] for seed in range(3)})
        registry=pd.DataFrame({'pattern_id':patterns,'role':np.where(part=='unknown','unknown','known')})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'repo';root.mkdir();data=Path(tmp)/'data';data.mkdir();(root/'outputs/results').mkdir(parents=True)
            with patch.object(oe,'ROOT',root),patch('src.common.config.DATA_DIR',data),patch.object(oe,'code_hashes',return_value={}), \
                 patch.object(oe,'load_inputs',return_value=(frame,registry,splits,known,{},{})), \
                 patch.object(oe,'split_validation',return_value=part), \
                 patch.object(oe.subprocess,'check_output',return_value='fake\n'), \
                 patch.object(oe,'run_classification',side_effect=RuntimeError('intentional stop')):
                with self.assertRaisesRegex(RuntimeError,'intentional'):oe.run_batch(run_id='synthetic')
            import json
            m=json.loads((root/'outputs/results/synthetic/manifest.json').read_text(encoding='utf-8'))
            self.assertEqual((m['complete'],m['completed_runs'],m['completed_policies']),(False,0,0))
            with patch.object(oe,'ROOT',root),patch('src.common.config.DATA_DIR',data):
                with self.assertRaises(FileExistsError):oe.run_batch(run_id='synthetic')


if __name__=='__main__':unittest.main()
