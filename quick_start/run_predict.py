import sys
sys.path.append('.')
try:
    from uie_wrapper import UIEPredictor
except Exception:
    from src import UIEPredictor


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--model", type=str, default='uie-base', help="The model to be used.")
    parser.add_argument("--task_path", type=str, default=None)
    parser.add_argument("--position_prob", default=0.5, type=float,
                        help="Probability threshold for start/end index probabiliry.")
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--device", choices=['cpu', 'gpu'], default="gpu")
    parser.add_argument("--schema_lang", choices=['zh', 'en'], default="zh")
    parser.add_argument("--engine", choices=['pytorch', 'onnx'], default="pytorch")
    parser.add_argument("--use_fp16", action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    python quick_start/run_predict.py \
        --model source_dir/pretrained/torch/uie-medical-base \
        --task_path source_dir/pretrained/torch/uie-medical-base \
        --max_seq_len 128 \
        --schema_lang zh \
        --engine pytorch \
    """
    import rich
    args = parse_args()
    schema = {
        'NER': ['疾病', '症状', '药物', '检查', '检验', '人名', '患者'],
        'Rel': {
            "患者": ['疾病', '症状', '药物', '检查']
        },
    }
    uie = UIEPredictor(
        model=args.model,
        task_path=args.task_path,
        schema=schema['NER'],
        engine=args.engine,
        device=args.device,
        position_prob=args.position_prob,
        max_seq_len=args.max_seq_len,
        batch_size=64,
        split_sentence=False,
        use_fp16=args.use_fp16
    )
    rich.print(uie("阿飞尿酸高怎么办, 老起能荨麻疹能不能吃莲花清瘟胶囊"))
    uie.set_schema(schema['Rel'])
    rich.print(uie("阿飞尿酸高怎么办, 能不能吃莲花清瘟胶囊"))
