from flask import Flask, jsonify, request, render_template
import time
import os
import base64
from PIL import Image
import matplotlib.pyplot as plt
import io
from models import OnnxModel, LlamaCPPLLM
import redis
import imagehash
import json

# load computer vision model
with open("./imagenet_classes.txt", "r") as f:
    labels = f.readlines()

resnet_model = OnnxModel("./mobilenetv2.onnx", labels)
llm = LlamaCPPLLM(
    "/app/models/models--microsoft--Phi-3-mini-4k-instruct-gguf/snapshots/c80d904a71b99a3eaeb8d3dbf164166384c09dc3/Phi-3-mini-4k-instruct-q4.gguf"
)
# llm = None


def base642img(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    return img


# app def
app = Flask(__name__)
# might need to change this when using the dockerfile
# redis_client = redis.Redis(host="localhost", port=6379)
redis_client = redis.Redis(host="redis", port=6379)


def generate_pred_graphic(img, predictions):
    plt.rc("font", size=25)
    plt.gca().set_position((0, 0, 1, 1))

    # reverse order of both labels and predictions
    labels = [item["label"] for item in predictions][::-1]
    predictions = [item["probability"] for item in predictions]
    predictions = [round(100 * p, 1) for p in predictions][::-1]

    labels = [f"{lab} ({p}%)" for lab, p in zip(labels, predictions)]

    fig, axs = plt.subplots(1, 2, figsize=(30, 10))
    axs[0].imshow(img, aspect="auto")
    hbars = axs[1].barh(range(len(predictions)), predictions, align="center")

    axs[1].bar_label(hbars, labels, padding=10)
    # axs[1].set_yticks(range(len(predictions)), labels=labels)
    # style
    axs[0].axis("off")
    axs[0].get_xaxis().set_ticks([])
    axs[0].get_yaxis().set_ticks([])

    axs[1].get_xaxis().set_ticks([])
    axs[1].get_yaxis().set_ticks([])
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].spines["bottom"].set_visible(False)
    axs[1].spines["left"].set_visible(False)

    return fig


@app.route("/predict/", methods=["POST", "GET"])
def do_infer():
    if request.method == "GET":
        # return file names to render
        context = {"objects": [f for f in os.listdir("./static/images/")]}
        return render_template("images.html", context=context)

    elif request.method == "POST":
        request_context = request.get_json()
        data = request_context.pop("data")
        img = base642img(data)

        hash = str(imagehash.average_hash(img)) + f'k{request_context.get("k",1)}'
        key = "image:" + hash

        cache = redis_client.get(key)
        if cache:
            # cache hit
            cache = json.loads(cache)
            return {**cache, "source": "cache"}

        # cache miss
        prediction = resnet_model.predict(img, **request_context)
        redis_client.set(key, json.dumps(prediction))

        # expire
        # redis_client.expire(key, 5)

        # save to file
        fig = generate_pred_graphic(img, prediction["model_outputs"])
        fig.savefig(
            f"./static/images/{hash}",
            transparent="True",
            pad_inches=0,
            bbox_inches="tight",
        )

        return jsonify({**prediction, "source": "API"})


@app.route("/chat/completions/", methods=["POST", "GET"])
def do_chat():
    if request.method == "POST":
        request_context = request.get_json()
        chat_context = {
            "messages": [
                {"role": "user", "content": "hey!"},
            ],
            "temperature": 0.8,
            "max_tokens": 256,
        }
        chat_context.update(request_context)

        partial_key = json.dumps(chat_context["messages"])
        key = "chat:" + partial_key
        cache = redis_client.get(key)
        if cache:
            # convo_pre = json.loads(partial_key)
            latest_response = json.loads(redis_client.get(key))
            return latest_response

        start = time.time()
        outputs = llm.generate(**chat_context)
        stop = time.time()
        outputs["speed"] = round(
            outputs["usage"]["completion_tokens"] / (stop - start), 2
        )

        redis_client.set(key, json.dumps(outputs))

        return jsonify(outputs)

    elif request.method == "GET":
        # context = {"objects": ["nate", "ben", "jack"]}
        context = {}
        keys = redis_client.keys("*chat*")
        values = [json.loads(redis_client.get(k)) for k in keys]
        # context["objects"] = values

        # decode the keys now since they contain the rest of the cover
        keys = [json.loads(k[5:]) for k in keys]
        chats = [k + [v["choices"][0]["message"]] for k, v in zip(keys, values)]

        speeds = [v["speed"] for v in values]

        # add fields here
        context["objects"] = [
            {"chat_history": c, "speed": s} for c, s in zip(chats, speeds)
        ]

        return render_template("chat.html", context=context)


if __name__ == "__main__":
    app.run("0.0.0.0", port=4440, debug=True)
