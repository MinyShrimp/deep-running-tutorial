import pandas as pd
from transformers import pipeline
from . import mps_device

classifier = pipeline(
    "text-classification",
    device=mps_device
)

outputs = classifier("We are very happy to show you the 🤗 Transformers library.")
print(outputs)
outputs_pd = pd.DataFrame(outputs)
print(outputs_pd)
