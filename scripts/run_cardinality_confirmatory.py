#!/usr/bin/env python3
"""Evaluate confirmatory cardinality policies and package every case."""

import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drill_ir.analysis.cardinality_confirmatory import apply_confirmatory_case, evaluate_confirmatory_case  # noqa:E402
from drill_ir.data import load_queries  # noqa:E402
from drill_ir.experiment import package_submission  # noqa:E402


def _hash(path):
    digest=hashlib.sha256();
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def _rankings(path):
    result={}
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        row=json.loads(line); qid=int(row['qid'])
        result[qid]=(list(map(int,row['retrieved_aids'])),list(map(float,row['scores'])))
    return result


def _write(path,selected,diagnostics):
    with path.open('w') as stream:
        for qid in sorted(selected): stream.write(json.dumps({'qid':qid,'retrieved_aids':selected[qid],**diagnostics[qid]})+'\n')


def _model_payload(fitted):
    return {
        'query_scaler':{'mean':fitted['query_scaler'].mean_.tolist(),'scale':fitted['query_scaler'].scale_.tolist()},
        'multiclass':{'classes':fitted['multiclass'].classes_.tolist(),'coefficient':fitted['multiclass'].coef_.tolist(),'intercept':fitted['multiclass'].intercept_.tolist()},
        'utility_scaler':{'mean':fitted['utility_scaler'].mean_.tolist(),'scale':fitted['utility_scaler'].scale_.tolist()},
        'ridge':{'coefficient':np.asarray(fitted['ridge'].coef_).tolist(),'intercept':float(fitted['ridge'].intercept_)},
        'utility_matrix':fitted['utility_matrix'].tolist(),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','dev-oof-predictions','train-queries','dev-qids','public-predictions','private-predictions','public-test','private-test','output-dir'):
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args(); config=yaml.safe_load(args.config.read_text())
    qids=set(map(int,json.loads(args.dev_qids.read_text())))
    gold={q.qid:q.relevant_aids for q in load_queries(args.train_queries) if q.qid in qids}; dev=_rankings(args.dev_oof_predictions)
    if set(gold)!=qids or set(dev)!=qids: raise ValueError('Development alignment failed')
    tests={'public':(_rankings(args.public_predictions),args.public_test),'private':(_rankings(args.private_predictions),args.private_test)}
    for name,(rankings,path) in tests.items():
        if set(rankings)!={q.qid for q in load_queries(path)}: raise ValueError(f'{name} alignment failed')
    args.output_dir.mkdir(parents=True,exist_ok=True); summary=[]
    inputs=[args.config,args.dev_oof_predictions,args.train_queries,args.dev_qids,args.public_predictions,args.private_predictions,args.public_test,args.private_test]
    for case in config['cases']:
        print(f'[confirmatory] start {case}',flush=True)
        metrics,deployment,fitted=evaluate_confirmatory_case(case,dev,gold,outer_folds=int(config['outer_folds']),inner_folds=int(config['inner_folds']),seed=int(config['seed']),maximum_k=int(config['maximum_k']),C=float(config['multiclass']['C']),alpha=float(config['ridge']['alpha']),lambda_values=list(map(float,config['ensemble']['lambda_multiclass'])))
        directory=args.output_dir/case; directory.mkdir(parents=True,exist_ok=True)
        (directory/'nested_oof_metrics.json').write_text(json.dumps(metrics,indent=2)); (directory/'deployment_selection.json').write_text(json.dumps(deployment,indent=2)); (directory/'model.json').write_text(json.dumps(_model_payload(fitted),indent=2))
        for split,(rankings,path) in tests.items():
            selected,diagnostics=apply_confirmatory_case(case,fitted,rankings,int(config['maximum_k']),deployment)
            prediction=directory/f'{split}_predictions.jsonl'; _write(prediction,selected,diagnostics)
            package_submission(prediction,set(rankings),directory/f'{split}_{case}_results.zip',output_k=int(config['maximum_k']))
        provenance={'case':case,'config':config,'nested_oof':metrics,'deployment':deployment,'input_sha256':{str(path):_hash(path) for path in inputs},'private_labels_used':False}
        (directory/'provenance.json').write_text(json.dumps(provenance,indent=2))
        summary.append({'case':case,**{key:metrics[key] for key in ('precision','recall','f2')},'mean_output_size':metrics['output_size']['mean'],'deployment':deployment})
        print(f"[confirmatory] complete {case} f2={metrics['f2']:.6f}",flush=True)
    summary.sort(key=lambda row:(row['f2'],row['precision'],row['recall']),reverse=True)
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)); (args.output_dir/'selected_by_nested_oof.json').write_text(json.dumps(summary[0],indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()

