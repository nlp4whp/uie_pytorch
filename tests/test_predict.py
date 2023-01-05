import pytest
try:
    from uie_wrapper.predictor import UIEPredictor
except Exception:
    from src.predictor import UIEPredictor
from ._helper import debug_logging


model_path = 'source_dir/pretrained/torch/uie-medical-base'
schema = {
    'NER': ['疾病', '症状', '药物', '检查', '检验', '人名', '患者'],
    'Rel': {
        "患者": ['疾病', '症状', '药物', '检查']
    },
}


@pytest.mark.parametrize('q', ["阿飞尿酸高怎么办, 老起能荨麻疹能不能吃莲花清瘟胶囊"])
def test_query_entity(q):
    uie = UIEPredictor(
        model=model_path,
        task_path=model_path,
        schema=schema['NER'],
        engine='pytorch',
        device='gpu',
        split_sentence=False,
        use_fp16=True
    )
    debug_logging(list(uie([q])))


@pytest.mark.parametrize('q', ["阿飞尿酸高怎么办, 能不能吃莲花清瘟胶囊"])
def test_query_triplet(q):
    uie = UIEPredictor(
        model=model_path,
        task_path=model_path,
        schema=schema['Rel'],
        engine='pytorch',
        device='gpu',
        split_sentence=False,
        use_fp16=True
    )
    debug_logging(list(uie([q])))
