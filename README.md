# simple-ml-api 

<div style="text-align: center;">
    <img src="https://github.com/nnethercott/nnethercott.github.io/blob/main/src/media/phi3_post/%20convo_demo.png?raw=true" style="width: 100%; display: block; margin: 0 auto;">
</div>

This repo shows how to setup a flask-based REST API serving a CNN for image classification, and Phi-3 for LLM chat. I originally made this to learn how to efficiently serve models using libraries like onnxruntime on my CPU-only machine.

Before launching the inference server make sure you have the model weights downloaded. To generate the ONNX file for mobilenetv2 run `models.py`. To get the gguf model weights for Phi-3 run the below:

```bash 
pip install huggingface-hub[cli]
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf --include Phi-3-mini-4k-instruct-q4.gguf
```

## Routes 
The project supports two main endpoints:
* `/chat/completions`: GET/POST requests for interacting with Phi-3
* `/predict`: GET/POST requests for image classification with mobilenetv2


<div style="text-align: center;">
    <img src="https://github.com/nnethercott/nnethercott.github.io/blob/main/src/media/phi3_post/rest_api_mobilenet.png?raw=true" style="width: 100%; display: block; margin: 0 auto;">
</div>
