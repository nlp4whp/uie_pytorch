import sys
sys.path.append('.')
try:
    from uie_wrapper.convert import download_and_transform
except Exception:
    from src.convert import download_and_transform


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--models", type=str, nargs="*", default=['uie-base', 'uie-medical-base'])
    parser.add_argument("--src_root", type=str, default='source_dir/pretrained/paddlenlp')
    parser.add_argument("--tgt_root", type=str, default='source_dir/temp')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    python quick_start/run_convert.py --models uie-base uie-123
    """
    args = parse_args()
    download_and_transform(
        args.src_root,
        args.tgt_root,
        model_names=args.models
    )
