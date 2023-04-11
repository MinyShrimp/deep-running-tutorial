# import pandas as pd
# from transformers import pipeline
# from . import mps_device
#
# classifier = pipeline(
#     "text-classification",
#     device=mps_device
# )
#
# outputs = classifier("We are very happy to show you the 🤗 Transformers library.")
# print(outputs)
# outputs_pd = pd.DataFrame(outputs)
# print(outputs_pd)

from transformers import AutoTokenizer
from bertviz.transformers_neuron_view import BertModel
from bertviz.neuron_view import show

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = BertModel.from_pretrained(model_ckpt)
text = "time files like an arrow"
show(model, "bert", tokenizer, text, display_mode="light", layer=0, head=8)
