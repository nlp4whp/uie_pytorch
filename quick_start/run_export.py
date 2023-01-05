import sys
sys.path.append('.')
from pathlib import Path
try:
    from uie_wrapper.export_model import export_pipeline
except Exception:
    from src.export_model import export_pipeline


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--onnx", type=str, default='model.onnx')
    parser.add_argument("--multilingual", type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    python quick_start/run_export.py \
        --model_path source_dir/temp/uie-medical-base \
        --onnx model.onnx \
    """
    args = parse_args()
    export_pipeline(
        args.model_path,
        args.onnx,
        multilingual=args.multilingual
    )
