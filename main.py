import os
import platform
from multiprocessing.pool import ThreadPool

import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

from xlsx.ReadNewsDataset import load_comment_contents

# init values
if platform.system() == "Windows":
    device = torch.device("cpu")
elif platform.system() == "Darwin":
    device = torch.device("mps")
else:   # Linux
    device = torch.device("cuda")

comment_df = load_comment_contents()

model_names = [
    "jaehyeong/koelectra-base-v3-generalized-sentiment-analysis",
    # "matthewburke/korean_sentiment",
    # "circulus/koelectra-sentiment-v1",
    "nlp04/korean_sentiment_analysis_dataset3",
    # "nlp04/korean_sentiment_analysis_kcelectra",
    # "nlp04/korean_sentiment_analysis_dataset3_best",
]

length = len(model_names)
pool = ThreadPool(processes=length)


def get_classifier(model_name: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device
    )


def run_classifier(_classifier):
    results = []
    df_slice = comment_df.iloc[:10]
    for idx, row in df_slice.iterrows():
        pred = _classifier(row["comment"])

        results.append({
            "id": idx,
            "result_out": pred[0]["label"],
            "result_score": pred[0]["score"],
        })
        print(f'{row["comment"]}\n>> {pred[0]}')

    return pd.DataFrame(results).set_index('id').columns(["result_out", "result_score"])


if __name__ == '__main__':
    # Model Load
    classifiers = [
        get_classifier(model_name) for model_name in model_names
    ]

    # Thread Create
    threads = [
        pool.apply_async(
            func=run_classifier,
            args=(classifier,)
        ) for classifier in classifiers
    ]

    # Thread Join & Get Result
    binary_results = [thread.get() for thread in threads[:1]]
    binary_result_df = comment_df.copy()
    for br in binary_results:
        binary_result_df.join(br)

    sentimental_results = [thread.get() for thread in threads[1:]]
    sentimental_result_df = comment_df.copy()
    for sr in sentimental_results:
        sentimental_result_df.join(sr)

    print(binary_result_df)
    print(sentimental_result_df)

    # Save Result
    save_file_name = os.path.join(os.getcwd(), "data", "result.xlsx")
    with pd.ExcelWriter(save_file_name, engine='openpyxl') as writer:
        binary_result_df.to_excel(writer, sheet_name="긍부정 결과")
        sentimental_result_df.to_excel(writer, sheet_name="감정 결과")
