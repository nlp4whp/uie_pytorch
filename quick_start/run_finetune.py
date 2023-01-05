import os
import sys
sys.path.append('.')
try:
    from uie_wrapper import do_train
except Exception:
    from src import do_train


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_dir", type=str, default="source_dir/temp/uie-base")
    parser.add_argument("--output_dir", type=str, default='./checkpoint')
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_model_num", type=int, default=5)
    parser.add_argument("--early_stopping", action='store_true')
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--valid_steps", type=int, default=500)
    parser.add_argument('--n_gpu', type=str, default="-1")
    parser.add_argument("--seed", default=123456, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    """
    python quick_start/run_finetune.py \
        --pretrained_dir source_dir/pretrained/torch/uie-base \
        --output_dir source_dir/temp/uie-demo-128 \
        --input_dir source_dir/dataset/demo_2022 \
        --max_seq_len 128 \
        --batch_size 32 \
        --num_epochs 100 \
        --learning_rate 1e-5 \
        --max_model_num 5 \
        --n_gpu 1 \
        --early_stopping \
        --logging_steps 100 \
        --valid_steps 500 \
        --seed 123456 \
    """

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.n_gpu

    do_train(
        pretrained_dir=args.pretrained_dir,
        output_dir=args.output_dir,
        train_data_path=os.path.join(args.input_dir, 'train.txt'),
        dev_data_path=os.path.join(args.input_dir, 'dev.txt'),
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        early_stopping=args.early_stopping,
        logging_steps=args.logging_steps,
        valid_steps=args.valid_steps,
        max_model_num=args.max_model_num,
        seed=args.seed,
        device="gpu" if int(os.environ['CUDA_VISIBLE_DEVICES']) >= 0 else "cpu",
        show_bar=False
    )
