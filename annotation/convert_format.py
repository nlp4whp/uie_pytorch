import rich
import json


def convert(from_files: list, to_file: str):
    """ demo数据格式转换为doccano支持的实体关系格式
    - demo-fmt: {'tokens': List[str], 'entity': [{'name': str, 'start': int, 'end': int, 'type': str}]}
    - doccano: {'id': int, 'text': str, 'entities': List, 'relations': List}
    """

    def _gen_item():
        for fn in from_files:
            rich.print("  open from file: ", fn)
            with open(fn) as fr:
                for line in fr:
                    yield json.loads(line)

    # rela_offset = 0
    entity_offset = 0
    with open(to_file, 'w') as fw:
        rich.print("write to file: ", to_file)
        for item_id, item in enumerate(_gen_item()):
            entities = [
                {
                    'id': ei + entity_offset,
                    'label': ett['type'],
                    'start_offset': ett['start'],
                    'end_offset': ett['end']
                } for ei, ett in enumerate(item['entity'])
            ]
            # rich.print(len(entities), type(entities), type(entities[0]))
            # raise Exception
            entity_offset += len(entities)
            relations = []  # {'id': int, 'from_id': int, 'to_id': int, 'type': str}
            fw.write(json.dumps({
                'id': item_id,
                'text': ''.join(item['tokens']),
                'relations': relations,
                'entities': entities
            }, ensure_ascii=False) + '\n')


def init_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_files", default=['train.rawjson'], type=str, nargs="*")
    parser.add_argument("--to_file", type=str, default="foo.jsonl")
    return parser.parse_args()


if __name__ == '__main__':

    """
    python convert_format.py  \
        --to_file demo-data/doccano.jsonl \
        --from_files demo-data/train.rawjson demo-data/eval.rawjson demo-data/test.rawjson \
    """
    args = init_args()
    convert(args.from_files, args.to_file)
