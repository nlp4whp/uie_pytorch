import os
from .utils import logger


class ONNXInferBackend(object):
    def __init__(self, model_path_prefix, device='cpu', use_fp16=False):
        from onnxruntime import InferenceSession, SessionOptions
        logger.info(">>> [ONNXInferBackend] Creating Engine ...")
        onnx_model = float_onnx_file = os.path.join(
            model_path_prefix, "inference.onnx")
        if not os.path.exists(onnx_model):
            raise OSError(f'{onnx_model} not exists!')
        infer_model_dir = model_path_prefix

        if device == "gpu":
            providers = ['CUDAExecutionProvider']
            logger.info(">>> [ONNXInferBackend] Use GPU to inference ...")
            if use_fp16:
                logger.info(">>> [ONNXInferBackend] Use FP16 to inference ...")
                from onnxconverter_common import float16
                import onnx
                fp16_model_file = os.path.join(infer_model_dir, "fp16_model.onnx")
                onnx_model = onnx.load_model(float_onnx_file)
                trans_model = float16.convert_float_to_float16(
                    onnx_model, keep_io_types=True)
                onnx.save_model(trans_model, fp16_model_file)
                onnx_model = fp16_model_file
        else:
            providers = ['CPUExecutionProvider']
            logger.info(">>> [ONNXInferBackend] Use CPU to inference ...")

        sess_options = SessionOptions()
        self.predictor = InferenceSession(
            onnx_model, sess_options=sess_options, providers=providers)
        if device == "gpu":
            try:
                assert 'CUDAExecutionProvider' in self.predictor.get_providers()
            except AssertionError:
                raise AssertionError(
                    "The environment for GPU inference is not set properly. "
                    "A possible cause is that you had installed both onnxruntime and onnxruntime-gpu. "
                    "Please run the following commands to reinstall: \n "
                    "1) pip uninstall -y onnxruntime onnxruntime-gpu \n 2) pip install onnxruntime-gpu"
                )
        logger.info(">>> [InferBackend] Engine Created ...")

    def infer(self, input_dict: dict):
        result = self.predictor.run(None, dict(input_dict))
        return result


class PyTorchInferBackend:
    def __init__(self, model_path_prefix, multilingual=False, device='cpu', use_fp16=False):
        from .uie import UIE, UIEM
        logger.info(">>> [PyTorchInferBackend] Creating Engine ...")
        if multilingual:
            self.model = UIEM.from_pretrained(model_path_prefix)
        else:
            self.model = UIE.from_pretrained(model_path_prefix)
        self.model.eval()
        self.device = device
        if self.device == 'gpu':
            logger.info(">>> [PyTorchInferBackend] Use GPU to inference ...")
            if use_fp16:
                logger.info(
                    ">>> [PyTorchInferBackend] Use FP16 to inference ...")
                self.model = self.model.half()
            self.model = self.model.cuda()
        else:
            logger.info(">>> [PyTorchInferBackend] Use CPU to inference ...")
        logger.info(">>> [PyTorchInferBackend] Engine Created ...")

    def infer(self, input_dict):
        import torch
        for input_name, input_value in input_dict.items():
            input_value = torch.LongTensor(input_value)
            if self.device == 'gpu':
                input_value = input_value.cuda()
            input_dict[input_name] = input_value

        outputs = self.model(**input_dict)
        start_prob, end_prob = outputs[0], outputs[1]
        if self.device == 'gpu':
            start_prob, end_prob = start_prob.cpu(), end_prob.cpu()
        start_prob = start_prob.detach().numpy()
        end_prob = end_prob.detach().numpy()
        return start_prob, end_prob
