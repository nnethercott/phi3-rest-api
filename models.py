from torchvision import models, transforms as T
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
import torch
from abc import ABC
import time
import onnxruntime as rt
import numpy as np
from PIL import Image
from llama_cpp import Llama


def sigmoid(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def preprocess(img):
    pipe = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return pipe(img).unsqueeze(0).detach().cpu().numpy()


# def preprocess(img):
#     delta = (256 - 224) // 2
#     img = img.resize((256, 256)).crop((delta, delta, 256 - delta, 256 - delta))
#
#     assert img.size == (224, 224)
#
#     tensor_inputs = np.array(img)
#     tensor_inputs = tensor_inputs / 255.0
#     tensor_inputs = (tensor_inputs - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
#     tensor_inputs = tensor_inputs.swapaxes(0, -1)[np.newaxis, ...]  # [1,3,(shape)]
#
#     return tensor_inputs


class OnnxModel:
    def __init__(self, onnx_file, labels):
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess = rt.InferenceSession(
            onnx_file, sess_options, providers=["CPUExecutionProvider"]
        )
        self.labels = labels

    # TODO: separate this into inference and post processing
    def predict(self, img, **kwargs):
        img = preprocess(img)
        img = img.astype(np.float32)
        logits = self.sess.run(["output"], {"input": img})[0]

        probs = sigmoid(logits)

        outputs = {}

        # return top k
        k = kwargs.get("k", 1)
        topk = np.argsort(-probs)[0][: int(k)]
        print(topk.shape)
        preds = [
            {"label": self.labels[i], "probability": probs[0][i].item()} for i in topk
        ]

        return {**outputs, "model_outputs": preds}


# llms
class LLM(ABC):
    def generate(self, **kwargs):
        raise NotImplementedError


class LlamaCPPLLM(LLM):
    def __init__(self, gguf_file):
        self.llm = Llama(gguf_file)

    def generate(self, **kwargs):
        return self.llm.create_chat_completion(**kwargs)


# class HuggingFaceLLM(LLM):
#     """microsoft/Phi-3-mini-4k-instruct"""
#
#     def __init__(self):
#         model_id = "microsoft/Phi-3-mini-4k-instruct"
#         self.llm = AutoModelForCausalLM.from_pretrained(
#             model_id, trust_remote_code=True
#         )
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
#
#     def generate(self, **kwargs):
#         inputs = self.tokenizer.apply_chat_template(
#             kwargs.pop("messages"), return_tensors="pt", return_dict=True
#         )
#         outputs = self.llm.generate(**inputs, **kwargs)
#         decoded = self.tokenizer.batch_decode(
#             outputs[:, inputs["input_ids"].shape[1] :]
#         )[0]
#
#         # format openai compatible response
#         return {"model_output": decoded}


if __name__ == "__main__":
    model = models.mobilenet_v2(MobileNet_V2_Weights.IMAGENET1K_V2)
    x = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,  # model
        x,  # dummy input for tracing
        "mobilenetv2.onnx",  # out filename
        export_params=True,  # also saves the weights in model (otherwise i guess you have the graph alone)
        opset_version=12,  # opset
        do_constant_folding=True,  # combines some constants
        input_names=["input"],
        output_names=["output"],
    )
