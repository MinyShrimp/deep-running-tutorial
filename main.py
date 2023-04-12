import os
from multiprocessing.pool import ThreadPool

import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

from xlsx.ReadNewsDataset import load_comment_contents

# init values
mps_device = torch.device("mps")
comment_df = load_comment_contents()

model_names = [
    "jaehyeong/koelectra-base-v3-generalized-sentiment-analysis",
    "nlp04/korean_sentiment_analysis_dataset3",
    "nlp04/korean_sentiment_analysis_kcelectra",
    "nlp04/korean_sentiment_analysis_dataset3_best",
    "matthewburke/korean_sentiment",
    "monologg/koelectra-small-finetuned-sentiment",
    "circulus/koelectra-sentiment-v1",
]

length = len(model_names)
pool = ThreadPool(processes=length)


def get_classifier(model_name: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=mps_device
    )


def run_classifier(_classifier):
    results = []
    for idx, row in comment_df.iterrows():
        pred = _classifier(row["comment_content"])

        results.append({
            "news_id": idx[0],
            "comment_id": idx[1],
            "comment_original_id": str(row["comment_original_id"]),
            "comment_content": row["comment_content"],
            "label": pred[0]["label"],
            "score": pred[0]["score"],
        })
        print(f'{row["comment_content"]}\n>> {pred[0]}')

    return pd.DataFrame(results)


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
    result_dfs = [thread.get() for thread in threads]

    # Save Result
    save_file_name = os.path.join(os.getcwd(), "data", "result.xlsx")
    with pd.ExcelWriter(save_file_name, engine='openpyxl') as writer:
        for name, df in zip(model_names, result_dfs):
            clean_name = name.replace("/", "_")
            df.to_excel(writer, sheet_name=clean_name, index=False)
