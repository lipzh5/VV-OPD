#!/usr/bin/env python3
"""Neutral task reward interface for on-policy distillation.

Provide the task reward callback required by the training framework.
Rollout correctness for visual contrastive supervision is handled
separately by the reward-gating module.
"""
# GEO3K_RELEASE_DATA_SOURCE_ALIAS_V107: allow official hiyouga/geometry3k label; neutral reward unchanged.
def compute_score(data_source,solution_str,ground_truth,extra_info=None,**kwargs):
    if (((str(data_source) != 'geo3k_stepmix_math') and (str(data_source)!='geo3k')) and (str(data_source) != 'hiyouga/geometry3k')):
        raise ValueError(f'Expected Geo3K-only data_source, got {data_source!r}')
    e=extra_info if isinstance(extra_info,dict) else {}
    if e.get('reward_task') not in (None,'','math'):
        raise ValueError(f'Unexpected task in Geo3K QA OPD: {e.get("reward_task")!r}')
    return {'score':0.0,'acc':0.0}
