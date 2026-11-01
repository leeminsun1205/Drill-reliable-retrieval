"""Fold-safe non-prefix evidence-set selection over a frozen top-five ranking."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
import random
from typing import Any

import numpy as np

from drill_ir.evaluation import evaluate_predictions
from .prefix_utility import _prefix_f2, _prefix_features, _validate_ranking


Ranking = tuple[Sequence[int], Sequence[float]]


def _folds(qids: Iterable[int], folds: int, seed: int) -> dict[int, int]:
    values = sorted(set(map(int, qids))); random.Random(seed).shuffle(values)
    return {qid: index % folds for index, qid in enumerate(values)}


def _masks(maximum_k: int) -> list[np.ndarray]:
    result = []
    for size in range(1, maximum_k + 1):
        for indices in combinations(range(maximum_k), size):
            mask = np.zeros(maximum_k, dtype=np.float64); mask[list(indices)] = 1.0; result.append(mask)
    return result


def _query_features(scores: np.ndarray, maximum_k: int) -> np.ndarray:
    return _prefix_features(scores, 1, maximum_k)


def _candidate_features(scores: np.ndarray, index: int, maximum_k: int) -> np.ndarray:
    base = _query_features(scores, maximum_k)
    padded = np.pad(scores[:maximum_k], (0, max(0, maximum_k-len(scores))), constant_values=scores[min(len(scores),maximum_k)-1])
    before = padded[index-1]-padded[index] if index else 0.0
    after = padded[index]-padded[index+1] if index+1 < maximum_k else 0.0
    return np.concatenate([base,[index/maximum_k,padded[index],padded[0]-padded[index],before,after]])


def _subset_features(scores: np.ndarray, mask: np.ndarray, maximum_k: int) -> np.ndarray:
    padded = np.pad(scores[:maximum_k], (0, max(0, maximum_k-len(scores))), constant_values=scores[min(len(scores),maximum_k)-1])
    chosen = padded[mask.astype(bool)]
    return np.concatenate([_query_features(scores,maximum_k),mask,mask*padded,[mask.sum()/maximum_k,chosen.sum(),chosen.mean(),chosen.std(),chosen.min(),chosen.max()]])


def _oracle_k(aids, gold, maximum_k):
    utilities=np.asarray([_prefix_f2(aids,gold,k) for k in range(1,min(maximum_k,len(aids))+1)])
    return int(np.argmax(utilities))+1


def _fit_cardinality_binary(rankings,gold,qids,maximum_k,seed,C_multi,C_binary):
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    qx,qy,cx,cy=[],[],[],[]
    for qid in sorted(qids):
        aids,scores=_validate_ranking(qid,rankings[qid]); limit=min(maximum_k,len(aids)); relevant=set(map(int,gold[qid]))
        qx.append(_query_features(scores,maximum_k)); qy.append(_oracle_k(aids,relevant,maximum_k)-1)
        for index in range(limit): cx.append(_candidate_features(scores,index,maximum_k)); cy.append(int(aids[index] in relevant))
    qs=StandardScaler().fit(np.vstack(qx)); cs=StandardScaler().fit(np.vstack(cx))
    multi_y=np.asarray(qy); binary_y=np.asarray(cy)
    multi=(
        DummyClassifier(strategy='constant',constant=int(multi_y[0]))
        if len(np.unique(multi_y)) == 1
        else LogisticRegression(C=C_multi,class_weight='balanced',max_iter=3000,random_state=seed)
    ).fit(qs.transform(np.vstack(qx)),multi_y)
    binary=(
        DummyClassifier(strategy='constant',constant=int(binary_y[0]))
        if len(np.unique(binary_y)) == 1
        else LogisticRegression(C=C_binary,class_weight='balanced',max_iter=3000,random_state=seed)
    ).fit(cs.transform(np.vstack(cx)),binary_y)
    return {'query_scaler':qs,'multiclass':multi,'candidate_scaler':cs,'binary':binary}


def _predict_cardinality_binary(fitted,rankings,maximum_k):
    selected,diagnostics={},{}; classes=list(map(int,fitted['multiclass'].classes_))
    for qid in sorted(rankings):
        aids,scores=_validate_ranking(qid,rankings[qid]); limit=min(maximum_k,len(aids))
        probs=fitted['multiclass'].predict_proba(fitted['query_scaler'].transform(_query_features(scores,maximum_k)[None,:]))[0]
        cardinality=classes[int(np.argmax(probs))]+1; cardinality=min(cardinality,limit)
        x=np.vstack([_candidate_features(scores,index,maximum_k) for index in range(limit)])
        binary_probabilities=fitted['binary'].predict_proba(fitted['candidate_scaler'].transform(x))
        binary_classes=list(map(int,fitted['binary'].classes_))
        relevance=(
            binary_probabilities[:,binary_classes.index(1)]
            if 1 in binary_classes
            else np.zeros(limit,dtype=np.float64)
        )
        indices=sorted(range(limit),key=lambda i:(-float(relevance[i]),i))[:cardinality]; indices.sort()
        selected[qid]=[aids[i] for i in indices]
        diagnostics[qid]={'selected_k':cardinality,'selected_ranks':[i+1 for i in indices],'non_prefix':indices!=list(range(cardinality)),'candidate_probabilities':list(map(float,relevance))}
    return selected,diagnostics


def _fit_subset_utility(rankings,gold,qids,maximum_k,alpha):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    masks=_masks(maximum_k); rows=[]; targets=[]
    for qid in sorted(qids):
        aids,scores=_validate_ranking(qid,rankings[qid]); relevant=set(map(int,gold[qid]))
        for mask in masks:
            chosen=[aids[i] for i in np.flatnonzero(mask) if i<len(aids)]; tp=len(set(chosen)&relevant)
            precision=tp/len(chosen); recall=tp/len(relevant); denominator=4*precision+recall
            rows.append(_subset_features(scores,mask,maximum_k)); targets.append(5*precision*recall/denominator if denominator else 0.0)
    scaler=StandardScaler().fit(np.vstack(rows)); model=Ridge(alpha=alpha).fit(scaler.transform(np.vstack(rows)),np.asarray(targets))
    return {'scaler':scaler,'model':model,'masks':masks}


def _predict_subset_utility(fitted,rankings,maximum_k):
    selected,diagnostics={},{}
    for qid in sorted(rankings):
        aids,scores=_validate_ranking(qid,rankings[qid]); valid=[mask for mask in fitted['masks'] if not np.any(np.flatnonzero(mask)>=len(aids))]
        x=np.vstack([_subset_features(scores,mask,maximum_k) for mask in valid]); values=fitted['model'].predict(fitted['scaler'].transform(x)); best=int(np.argmax(values)); indices=list(map(int,np.flatnonzero(valid[best])))
        selected[qid]=[aids[i] for i in indices]; diagnostics[qid]={'selected_k':len(indices),'selected_ranks':[i+1 for i in indices],'non_prefix':indices!=list(range(len(indices))),'predicted_utility':float(values[best])}
    return selected,diagnostics


def _crossfit(case,param,rankings,gold,qids,folds,seed,maximum_k,C_multi):
    assignment=_folds(qids,folds,seed); predictions={}; diagnostics={}
    for fold in range(folds):
        train={qid for qid in qids if assignment[qid]!=fold}; validation=qids-train
        if case=='cardinality_binary_subset': fitted=_fit_cardinality_binary(rankings,gold,train,maximum_k,seed+fold,C_multi,param); selected,details=_predict_cardinality_binary(fitted,{qid:rankings[qid] for qid in validation},maximum_k)
        else: fitted=_fit_subset_utility(rankings,gold,train,maximum_k,param); selected,details=_predict_subset_utility(fitted,{qid:rankings[qid] for qid in validation},maximum_k)
        predictions.update(selected); diagnostics.update(details)
    return predictions,diagnostics


def _select_param(case,grid,rankings,gold,qids,folds,seed,maximum_k,C_multi):
    records=[]; best=None
    for index,param in enumerate(grid):
        predictions,_=_crossfit(case,param,rankings,gold,qids,folds,seed+1000,maximum_k,C_multi); metrics=evaluate_predictions({qid:gold[qid] for qid in qids},predictions).to_dict(); records.append({'parameter':param,'metrics':metrics})
        key=(metrics['f2'],metrics['precision'],metrics['recall'],-index)
        if best is None or key>best[0]: best=(key,param)
    return best[1],records


def evaluate_subset_case(case,rankings,gold,*,outer_folds,inner_folds,seed,maximum_k,C_multi,grid):
    qids=set(map(int,gold)); outer=_folds(qids,outer_folds,seed); predictions={}; diagnostics={}; folds=[]
    for fold in range(outer_folds):
        train={qid for qid in qids if outer[qid]!=fold}; validation=qids-train
        param,tuning=_select_param(case,grid,rankings,gold,train,inner_folds,seed+fold,maximum_k,C_multi)
        if case=='cardinality_binary_subset': fitted=_fit_cardinality_binary(rankings,gold,train,maximum_k,seed+fold,C_multi,param); selected,details=_predict_cardinality_binary(fitted,{qid:rankings[qid] for qid in validation},maximum_k)
        else: fitted=_fit_subset_utility(rankings,gold,train,maximum_k,param); selected,details=_predict_subset_utility(fitted,{qid:rankings[qid] for qid in validation},maximum_k)
        predictions.update(selected); diagnostics.update(details); folds.append({'fold':fold,'selected_parameter':param,'inner_tuning':tuning,'validation_metrics':evaluate_predictions({qid:gold[qid] for qid in validation},selected).to_dict()})
    metrics=evaluate_predictions(gold,predictions).to_dict(); sizes=np.asarray([len(v) for v in predictions.values()]); non_prefix=sum(bool(row['non_prefix']) for row in diagnostics.values())/len(diagnostics)
    metrics.update(output_size={'minimum':int(sizes.min()),'mean':float(sizes.mean()),'median':float(np.median(sizes)),'maximum':int(sizes.max())},non_prefix_rate=non_prefix,folds=folds)
    deployment_param,tuning=_select_param(case,grid,rankings,gold,qids,outer_folds,seed,maximum_k,C_multi)
    fitted=_fit_cardinality_binary(rankings,gold,qids,maximum_k,seed,C_multi,deployment_param) if case=='cardinality_binary_subset' else _fit_subset_utility(rankings,gold,qids,maximum_k,deployment_param)
    return metrics,{'selected_parameter':deployment_param,'cross_fitted_tuning':tuning},fitted,predictions,diagnostics


def apply_subset_case(case,fitted,rankings,maximum_k):
    return _predict_cardinality_binary(fitted,rankings,maximum_k) if case=='cardinality_binary_subset' else _predict_subset_utility(fitted,rankings,maximum_k)


def oracle_subset_metrics(rankings,gold,maximum_k):
    prefix={}; subset={}
    for qid in sorted(gold):
        aids,_=_validate_ranking(qid,rankings[qid]); relevant=set(map(int,gold[qid])); limit=min(maximum_k,len(aids)); k=_oracle_k(aids,relevant,maximum_k); prefix[qid]=aids[:k]
        hits=[aid for aid in aids[:limit] if aid in relevant]; subset[qid]=hits if hits else [aids[0]]
    return {'oracle_prefix':evaluate_predictions(gold,prefix).to_dict(),'oracle_subset':evaluate_predictions(gold,subset).to_dict()},prefix,subset

