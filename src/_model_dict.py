
MODEL_MAP = {
    # vocab.txt/special_tokens_map.json/tokenizer_config.json are common to the default model.
    "uie-base": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_v1.0/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/model_config.json",
            "vocab.txt":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json"
        }
    },
    "uie-medium": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medium_v1.0/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medium/model_config.json",
            "vocab.txt":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-mini": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_mini_v1.0/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_mini/model_config.json",
            "vocab.txt":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-micro": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_micro_v1.0/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_micro/model_config.json",
            "vocab.txt":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-nano": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_nano_v1.0/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_nano/model_config.json",
            "vocab.txt":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-medical-base": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medical_base_v0.1/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/model_config.json",
            "vocab.txt":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-base-en": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en_v1.1/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/model_config.json",
            "vocab.txt":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/vocab.txt",
            "special_tokens_map.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/special_tokens_map.json",
            "tokenizer_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/tokenizer_config.json",
        }
    },
    # uie-m模型需要Ernie-M模型
    "uie-m-base": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_base_v1.0/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_base/model_config.json",
            "vocab.txt":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_base/vocab.txt",
            "special_tokens_map.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_base/special_tokens_map.json",
            "tokenizer_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_base/tokenizer_config.json",
            "sentencepiece.bpe.model":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_base/sentencepiece.bpe.model"

        }
    },
    "uie-m-large": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_large_v1.0/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_large/model_config.json",
            "vocab.txt":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_large/vocab.txt",
            "special_tokens_map.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_large/special_tokens_map.json",
            "tokenizer_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_large/tokenizer_config.json",
            "sentencepiece.bpe.model":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_base/sentencepiece.bpe.model"
        }
    },
    # Rename to `uie-medium` and the name of `uie-tiny` will be deprecated in future.
    # "uie-tiny": {
    #     "resource_file_urls": {
    #         "model_state.pdparams":
    #         "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny_v0.1/model_state.pdparams",
    #         "model_config.json":
    #         "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/model_config.json",
    #         "vocab.txt":
    #         "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/vocab.txt",
    #         "special_tokens_map.json":
    #         "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/special_tokens_map.json",
    #         "tokenizer_config.json":
    #         "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/tokenizer_config.json"
    #     }
    # },
    "ernie-3.0-base-zh": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_base_zh.pdparams",
            "model_config.json":
            "base64:ew0KICAiYXR0ZW50aW9uX3Byb2JzX2Ryb3BvdXRfcHJvYiI6IDAuMSwNCiAgImhpZGRlbl9hY3QiOiAiZ2VsdSIsDQogICJoaWRkZW5fZHJvcG91dF9wcm9iIjogMC4xLA0KICAiaGlkZGVuX3NpemUiOiA3NjgsDQogICJpbml0aWFsaXplcl9yYW5nZSI6IDAuMDIsDQogICJtYXhfcG9zaXRpb25fZW1iZWRkaW5ncyI6IDIwNDgsDQogICJudW1fYXR0ZW50aW9uX2hlYWRzIjogMTIsDQogICJudW1faGlkZGVuX2xheWVycyI6IDEyLA0KICAidGFza190eXBlX3ZvY2FiX3NpemUiOiAzLA0KICAidHlwZV92b2NhYl9zaXplIjogNCwNCiAgInVzZV90YXNrX2lkIjogdHJ1ZSwNCiAgInZvY2FiX3NpemUiOiA0MDAwMCwNCiAgImluaXRfY2xhc3MiOiAiRXJuaWVNb2RlbCINCn0=",
            "vocab.txt":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_base_zh_vocab.txt",
            "special_tokens_map.json":
            "base64:eyJ1bmtfdG9rZW4iOiAiW1VOS10iLCAic2VwX3Rva2VuIjogIltTRVBdIiwgInBhZF90b2tlbiI6ICJbUEFEXSIsICJjbHNfdG9rZW4iOiAiW0NMU10iLCAibWFza190b2tlbiI6ICJbTUFTS10ifQ==",
            "tokenizer_config.json":
            "base64:eyJkb19sb3dlcl9jYXNlIjogdHJ1ZSwgInVua190b2tlbiI6ICJbVU5LXSIsICJzZXBfdG9rZW4iOiAiW1NFUF0iLCAicGFkX3Rva2VuIjogIltQQURdIiwgImNsc190b2tlbiI6ICJbQ0xTXSIsICJtYXNrX3Rva2VuIjogIltNQVNLXSIsICJ0b2tlbml6ZXJfY2xhc3MiOiAiRXJuaWVUb2tlbml6ZXIifQ=="
        }
    }
}
