import pytest
try:
    from uie_wrapper.export_model import export_pipeline
except Exception:
    from src.export_model import export_pipeline


CKPT_ROOT = 'source_dir/temp'


# @pytest.mark.skipif(
#     os.path.exists('source_dir/pretrained/paddlenlp/uie-base'),
#     reason='Warning: model path exists'
# )
@pytest.mark.parametrize('name', ['uie-base', 'uie-medical-base'])
def test_export_onnx(name):
    export_pipeline(f"{CKPT_ROOT}/{name}", 'pytest.onnx', multilingual=False)
