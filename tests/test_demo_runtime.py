"""Guard memory placement without changing the frozen retrieval treatment."""
from copy import deepcopy
import pytest

from drill_ir.demo.runtime import pin_dense_search_to_cpu


@pytest.mark.parametrize('device', ['auto', 'cuda:0', 'cuda:1', 'cpu'])
def test_search_device_overrides_profiles_and_preserves_scientific_parameters(device):
    retrieval = {
        'top_k': 100,
        'hybrid': {'rrf_k': 10, 'sparse_weight': 0.15, 'dense_components': [
            {'name': 'harrier', 'weight': 0.65}, {'name': 'e5', 'weight': 0.20},
        ]},
        'dense': {'search': {'backend': 'faiss_flat', 'device': device, 'metric': 'cosine'},
                  'encoder': {'model_name': 'unchanged'},
                  'chunking': {'method': 'token_window', 'max_tokens': 448, 'overlap_tokens': 64}},
        'dense_profiles': {
            'harrier': {'search': {'device': device, 'query_batch_size': 64}},
            'e5': {'encoder': {'model_name': 'e5'}},
        },
    }
    expected = deepcopy(retrieval)
    expected['dense']['search']['device'] = 'cpu'
    expected['dense_profiles']['harrier']['search']['device'] = 'cpu'
    expected['dense_profiles']['e5']['search'] = {'device': 'cpu'}
    devices = pin_dense_search_to_cpu(retrieval)
    assert retrieval == expected
    assert devices == {'base_dense': 'cpu', 'harrier': 'cpu', 'e5': 'cpu'}


def test_cpu_device_is_applied_even_without_explicit_search_mapping():
    retrieval = {'dense': {}, 'dense_profiles': {'one': {}}}
    pin_dense_search_to_cpu(retrieval)
    assert retrieval == {'dense': {'search': {'device': 'cpu'}},
                         'dense_profiles': {'one': {'search': {'device': 'cpu'}}}}


@pytest.mark.parametrize('free,expected', [
    ([3.52, 14.46], 'cuda:1'),
    ([0.35, 12.05], 'cuda:1'),
    ([14.0, 13.0], ['cuda:0', 'cuda:1']),
    ([12.0, 14.0], ['cuda:1', 'cuda:0']),
    ([8.0], 'cuda:0'),
    ([3.0], 'cpu'),
    ([3.0, 4.0], 'cpu'),
    ([], 'cpu'),
])
def test_reranker_placement_respects_per_device_headroom(free, expected):
    from drill_ir.demo.runtime import choose_reranker_device
    assert choose_reranker_device(free) == expected


def test_unloaded_reranker_keeps_model_and_scoring_contract():
    from dataclasses import dataclass
    from types import SimpleNamespace
    from drill_ir.demo.runtime import place_unloaded_reranker
    @dataclass(frozen=True)
    class Config:
        model_name: str = 'checkpoint-1234'
        max_length: int = 512
        batch_size: int = 8
        device: str = 'cuda:0'
        fp16: bool = True
    scorer = SimpleNamespace(_model=None, config=Config())
    assert place_unloaded_reranker(scorer, [3.52, 14.46]) == 'cuda:1'
    assert scorer.config == Config(device='cuda:1')
    assert place_unloaded_reranker(scorer, []) == 'cpu'
    assert scorer.config == Config(device='cpu', fp16=False)
    scorer._model = object()
    with pytest.raises(RuntimeError):
        place_unloaded_reranker(scorer, [14., 14.])
