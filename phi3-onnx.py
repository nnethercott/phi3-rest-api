import onnxruntime_genai as og 
import time

model = og.Model(f'Phi-3-mini-4k-instruct-onnx/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/')
tokenizer = og.Tokenizer(model) 

chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>' 
prompt = f'{chat_template.format(input="Tell me a funny joke please.")}'
input_tokens = tokenizer.encode(prompt)

params = og.GeneratorParams(model)
search_options = {"temperature": 0.2, "max_length": 100, "do_sample": True}
params.set_search_options(**search_options)
params.input_ids = input_tokens

generator = og.Generator(model, params)


start = time.time()
completion_tokens = []

while not generator.is_done():
  generator.compute_logits()
  generator.generate_next_token()
  completion_tokens.append(generator.get_next_tokens()[0])

stop = time.time()

outputs = tokenizer.decode(completion_tokens)
print(outputs)
print(f"{len(completion_tokens)} tokens generated in {stop-start} s.")
