import pytest
try:
    from uie_wrapper.convert import download_and_transform
except Exception:
    from src.convert import download_and_transform


SRC_ROOT = 'source_dir/pretrained/paddlenlp'
TGT_ROOT = 'source_dir/temp'
ALL_FILE = ["pytorch_model.bin", "config.json", "tokenizer_config.json", "vocab.txt"]


# @pytest.mark.skipif(
#     os.path.exists('source_dir/pretrained/paddlenlp/uie-base'),
#     reason='Warning: model path exists'
# )
@pytest.mark.parametrize('name', ['uie-base', 'uie-medical-base'])
def test_paddle_to_pytorch(name):
    download_and_transform(SRC_ROOT, TGT_ROOT, model_names=[name])
