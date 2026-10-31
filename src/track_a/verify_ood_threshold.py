"""저장 원본·확률 벡터로 문턱과 거절 지표를 별도 계산한다. 학습하지 않는다."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from src.common.models import make_model
from src.common.results import sha256_file, summarize_runs
from src.common.splits import split_validation
from src.track_a.feature_group_experiment import ROOT, load_inputs
from src.track_a.ood_threshold_experiment import code_hashes
from src.track_a.preprocess import feature_groups
from src.track_a.result_metadata import read_archive

UNKNOWN = "__unknown__"
FIELDS = ("macro_f1", "accuracy", "known_reject_rate", "unknown_reject_rate",
          "unknown_class_macro_reject_rate", "benign_reject_rate", "benign_wrong_known_rate",
          "accepted_known_accuracy", "known_coverage", "unknown_pattern_macro_reject_rate")


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0, equal_nan=True)


def verify(run_id, data_dir, machine):
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("배치 ID 오류")
    out = ROOT / "outputs/results" / run_id
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if (not manifest["complete"] or manifest["completed_runs"] != 12 or manifest["completed_policies"] != 48
            or manifest["machine"] != machine or manifest["code_sha256"] != code_hashes()):
        raise ValueError("완료·코드·환경 불일치")
    frame, rows, splits, known, paths, hashes = load_inputs(data_dir, manifest["preprocess_run_id"])
    if hashes != manifest["source_sha256"] or known != manifest["known_labels"]:
        raise ValueError("입력 또는 클래스 불일치")
    archive = read_archive(out)
    if archive is None or len(archive) != 12:
        raise ValueError("메타데이터12건 필요")
    read = lambda name: pd.read_csv(out / f"{name}.csv", float_precision="round_trip")
    metrics, confusion, thresholds, policies, opened, classes, patterns, pattern_summary, audit, scores, auc, summary, paired = map(read,
        ("metrics", "confusion", "thresholds", "policy_metrics", "open_confusion", "class_rejection", "pattern_rejection",
         "pattern_summary", "partition_audit", "score_audit", "auc", "policy_summary", "paired"))
    sizes = [12, 2028, 36, 48, 9408, 864, None, 240, 288, 36, 12, 16, 36]
    for table, size in zip((metrics, confusion, thresholds, policies, opened, classes, patterns, pattern_summary,
                           audit, scores, auc, summary, paired), sizes):
        if size is not None and len(table) != size:
            raise ValueError(f"건수 불일치: {len(table)} != {size}")
    wanted = {(s, seed, f) for s in ("random", "pattern_group") for seed in range(3) for f in ("all", "sequence")}
    if len(metrics) != len(wanted) or set(map(tuple, metrics[["split", "seed", "feature_group"]].to_numpy())) != wanted:
        raise ValueError("조건 누락/중복")
    # Independent literal byte representation and hash of the original signed sequence.
    raw = frame[[str(i) for i in range(30)]].to_numpy(dtype="<i8")
    literal = pd.Series([x.tobytes().hex() for x in raw], index=frame.index)
    literal_to_hash = {x: hashlib.sha256(bytes.fromhex(x)).hexdigest() for x in literal.unique()}
    if not np.array_equal(literal.map(literal_to_hash), rows.pattern_id):
        raise ValueError("원본 패턴 해시 오류")
    partitions = {}
    for split in ("random", "pattern_group"):
        for seed in range(3):
            outer = splits[f"{split}_{seed}"]
            part = split_validation(frame.Label, outer, split=split, seed=seed,
                                   groups=rows.pattern_id if split == "pattern_group" else None)
            partitions[split, seed] = part
            train = set(literal.loc[part.eq("train")])
            local_audit = audit.loc[audit.split.eq(split) & audit.seed.eq(seed)]
            for r in local_audit.to_dict("records"):
                mask = part.eq(r["partition"])
                if r["label"] != "__all__": mask &= frame.Label.eq(r["label"])
                values = literal.loc[mask]
                expected = dict(rows=len(values), unique_patterns=values.nunique(), train_patterns=len(train),
                    shared_patterns=len(set(values) & train), train_pattern_seen_rows=int(values.isin(train).sum()),
                    train_pattern_seen_fraction=float(values.isin(train).mean()) if len(values) else np.nan)
                for name, value in expected.items(): close(r[name], value)
    checked_rows = 0
    for r in metrics.to_dict("records"):
        child = f"{run_id}_{r['split']}_rf_{r['feature_group']}_s{r['seed']}"
        meta = archive[child]
        part = partitions[r["split"], r["seed"]]
        if (r["model"] != "rf" or r["machine"] != machine or r["validation_rows"] <= 0
                or meta["validation_refit"] is not False or meta["fit_partition"] != "train"
                or meta["probability_classes"] != known or meta["feature_columns"] != feature_groups()[r["feature_group"]]
                or meta["extra"]["code_sha256"] != manifest["code_sha256"]
                or meta["model_params"] != make_model("rf", seed=r["seed"], workers=2).get_params()):
            raise ValueError("실행 설정/검증 배정 오류")
        folder = data_dir / "predictions" / child
        for name in ("predictions.csv", "assignments.csv"):
            raw_file = (folder / name).read_bytes();record = meta["extra"]["row_files"][name]
            if (hashlib.sha256(raw_file).hexdigest() != record["sha256"] or len(raw_file) != record["bytes"]
                    or hashlib.sha256(raw_file.replace(b"\r\n", b"\n")).hexdigest() != record["lf_sha256"]):
                raise ValueError("원본 행별 파일 SHA/크기 불일치")
        p = pd.read_csv(folder / "predictions.csv", float_precision="round_trip")
        assign = pd.read_csv(folder / "assignments.csv", keep_default_na=False)
        assert assign.row_id.tolist() == frame.index.tolist() and assign.partition.tolist() == part.tolist()
        ids = part.index[part.isin(["validation", "test", "unknown"])]
        assert p.row_id.tolist() == ids.tolist() and p.partition.tolist() == part.loc[ids].tolist()
        assert p.actual.tolist() == frame.Label.loc[ids].tolist()
        columns = [f"prob_{i:02d}" for i in range(13)]
        assert meta["probability_columns"] == columns
        assert set(p.columns) == {"row_id", "partition", "actual", "predicted", "max_probability", *columns}
        prob = p[columns].to_numpy();assert np.isfinite(prob).all() and ((0 <= prob) & (prob <= 1)).all()
        close(prob.sum(1), 1)
        np.testing.assert_array_equal(prob.max(1), p.max_probability)
        np.testing.assert_array_equal(np.array(known)[prob.argmax(1)], p.predicted)
        checked_rows += len(p)
        val = p.loc[p.partition.eq("validation"), "max_probability"].to_numpy()
        point = {"none": 0.}
        saved = thresholds.loc[thresholds.source_run_id.eq(child)]
        assert set(saved.policy_id) == {"reject_01", "reject_05", "reject_10"}
        for percent in (1, 5, 10):
            budget = len(val) * percent // 100
            tau = float(np.sort(val)[budget])  # Independent order statistic, without quantile interpolation.
            row = saved.loc[saved.policy_id.eq(f"reject_{percent:02d}")].iloc[0]
            assert row.threshold == tau
            expected = dict(target_reject_rate=percent / 100, validation_rows=len(val), allowed_reject_rows=budget,
                validation_reject_rows=int((val < tau).sum()), validation_reject_rate=float((val < tau).mean()),
                boundary_tie_rows=int((val == tau).sum()), unique_scores=len(set(val)))
            for name, value in expected.items(): close(row[name], value)
            point[f"reject_{percent:02d}"] = tau
        stored_calibration = {x['policy_id']: x for x in meta['extra']['calibration']}
        for _, row in saved.iterrows():
            for name in ('threshold','validation_rows','allowed_reject_rows','validation_reject_rows','boundary_tie_rows'):
                close(stored_calibration[row.policy_id][name],row[name])
        local_scores = scores.loc[scores.source_run_id.eq(child)]
        assert set(local_scores.partition) == {"validation", "test", "unknown"}
        for _, row in local_scores.iterrows():
            values = p.loc[p.partition.eq(row.partition), "max_probability"]
            expected = dict(rows=len(values), unique_scores=values.nunique(), minimum=values.min(), maximum=values.max(),
                            fraction_one=values.eq(1).mean())
            expected.update(dict(zip(("q01", "q05", "q50", "q95", "q99"), values.quantile([.01,.05,.5,.95,.99]).to_numpy())))
            for name, value in expected.items(): close(row[name], value)
        test = p.loc[p.partition.eq("test")]
        for name, value in dict(macro_f1=f1_score(test.actual,test.predicted,labels=known,average="macro",zero_division=0),
            accuracy=accuracy_score(test.actual,test.predicted), train_rows=int(part.eq("train").sum()),
            validation_rows=len(val),test_rows=len(test),unknown_rows=int(part.eq("unknown").sum())).items():close(r[name],value)
        expected_matrix = confusion_matrix(test.actual,test.predicted,labels=known)
        cm = confusion.loc[confusion.split.eq(r['split']) & confusion.seed.eq(r['seed']) & confusion.feature_group.eq(r['feature_group'])]
        np.testing.assert_array_equal(cm.pivot(index='actual',columns='predicted',values='count').reindex(index=known,columns=known),expected_matrix)
        e = p.loc[p.partition.ne('validation')].copy();is_known=e.partition.eq('test').to_numpy();truth=np.where(is_known,e.actual,UNKNOWN)
        arow=auc.loc[auc.source_run_id.eq(child)].iloc[0]
        close(arow.auroc,roc_auc_score(~is_known,1-e.max_probability))
        close(arow.known_rows,is_known.sum());close(arow.unknown_rows,(~is_known).sum())
        for policy, tau in point.items():
            rejected=e.max_probability.to_numpy()<tau;chosen=np.where(rejected,UNKNOWN,e.predicted);selected=is_known&~rejected
            actual=policies.loc[policies.source_run_id.eq(child)&policies.policy_id.eq(policy)];assert len(actual)==1;actual=actual.iloc[0]
            assert actual.threshold==tau
            benign=e.actual.eq('benign').to_numpy()&is_known
            expected=dict(rows=len(e),known_rows=int(is_known.sum()),unknown_rows=int((~is_known).sum()),
                macro_f1=f1_score(truth,chosen,labels=[*known,UNKNOWN],average='macro',zero_division=0),accuracy=accuracy_score(truth,chosen),
                known_reject_rows=int(rejected[is_known].sum()),unknown_reject_rows=int(rejected[~is_known].sum()),
                known_reject_rate=rejected[is_known].mean(),unknown_reject_rate=rejected[~is_known].mean(),
                accepted_known_rows=int(selected.sum()),known_coverage=selected.sum()/is_known.sum(),
                accepted_known_accuracy=accuracy_score(truth[selected],chosen[selected]) if selected.any() else np.nan,
                benign_rows=int(benign.sum()),benign_reject_rows=int(rejected[benign].sum()),benign_reject_rate=rejected[benign].mean(),
                benign_wrong_known_rows=int((benign&~rejected&(chosen!='benign')).sum()),
                benign_wrong_known_rate=(~rejected[benign]&(chosen[benign]!='benign')).mean())
            cr=classes.loc[classes.source_run_id.eq(child)&classes.policy_id.eq(policy)];rates=[]
            for row in cr.to_dict('records'):
                mask=e.partition.eq(row['partition']).to_numpy()&e.actual.eq(row['label']).to_numpy()
                close(row['rows'],mask.sum());close(row['reject_rows'],rejected[mask].sum());close(row['reject_rate'],rejected[mask].mean())
                assert row['low_support']==bool(0<mask.sum()<30)
                if row['partition']=='unknown':rates.append(rejected[mask].mean())
            expected['unknown_class_macro_reject_rate']=np.mean(rates)
            u=e.loc[~is_known,['row_id','actual']].copy();u['literal']=literal.loc[u.row_id].to_numpy();u['rejected']=rejected[~is_known].astype(int)
            grouped=u.groupby(['actual','literal']).rejected.agg(['size','sum']);grouped['rate']=grouped['sum']/grouped['size']
            pr=patterns.loc[patterns.source_run_id.eq(child)&patterns.policy_id.eq(policy)]
            wanted_patterns={(label,literal_to_hash[lit]) for label,lit in grouped.index}
            assert set(zip(pr.label,pr.pattern_id))==wanted_patterns and len(pr)==len(wanted_patterns)
            inverse={v:k for k,v in literal_to_hash.items()}
            for row in pr.to_dict('records'):
                g=grouped.loc[(row['label'],inverse[row['pattern_id']])]
                close(row['rows'],g['size']);close(row['reject_rows'],g['sum']);close(row['reject_rate'],g['rate'])
            per_class=grouped.groupby(level=0)['rate'].mean();expected['unknown_pattern_macro_reject_rate']=per_class.mean()
            ps=pattern_summary.loc[pattern_summary.source_run_id.eq(child)&pattern_summary.policy_id.eq(policy)]
            for row in ps.to_dict('records'):
                g=grouped.loc[row['label']];close(row['rows'],g['size'].sum());close(row['patterns'],len(g))
                close(row['pattern_equal_weight_reject_rate'],per_class.loc[row['label']])
            for name,value in expected.items():close(actual[name],value)
            cm=opened.loc[opened.source_run_id.eq(child)&opened.policy_id.eq(policy)];labs=[*known,UNKNOWN]
            np.testing.assert_array_equal(cm.pivot(index='actual',columns='predicted',values='count').reindex(index=labs,columns=labs),confusion_matrix(truth,chosen,labels=labs))
        print('verified',child,flush=True)
    for _,r in summary.iterrows():
        group=policies
        for key in ['machine','feature_group','split','model','policy_id']:group=group.loc[group[key].eq(r[key])]
        assert len(group)==3 and set(group.seed)=={0,1,2}
        for metric in FIELDS:
            close(r[metric+'_mean'],group[metric].mean());close(r[metric+'_std'],group[metric].std(ddof=1));close(r[metric+'_valid_runs'],group[metric].notna().sum())
    for _,r in paired.iterrows():
        group=policies.loc[policies.source_run_id.eq(r.source_run_id)];base=group.loc[group.policy_id.eq('none')].iloc[0];point=group.loc[group.policy_id.eq(r.policy_id)].iloc[0]
        for metric in FIELDS:close(r[metric],point[metric]);close(r[metric+'_none'],base[metric]);close(r['delta_'+metric],point[metric]-base[metric])
    pd.testing.assert_frame_equal(read('summary').fillna(''),summarize_runs(metrics).fillna(''),check_dtype=False,rtol=0,atol=1e-12)
    if code_hashes()!=manifest['code_sha256'] or any(sha256_file(path)!=hashes[name] for name,path in paths.items()):
        raise ValueError('검증 중 소스/입력 변경')
    report=dict(complete=True,run_id=run_id,machine=machine,models=12,policies=48,thresholds=36,
        probability_rows=checked_rows,open_confusion_cells=9408,closed_confusion_cells=2028,
        row_files=24,pattern_rows=len(patterns),verifier_sha256=sha256_file(Path(__file__)),
        aggregate_sha256={f.name:sha256_file(f) for f in out.iterdir() if f.is_file() and f.name!='verification.json'})
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('VERIFIED 12 models / 48 policies / full probability vectors',flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True);args=parser.parse_args()
    from src.common.config import DATA_DIR,MACHINE
    verify(args.run_id,DATA_DIR.expanduser().resolve(),MACHINE)


if __name__=='__main__':main()
