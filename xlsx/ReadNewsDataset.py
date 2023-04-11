import os
from pandas import read_excel


def get_full_path(filename: str):
    data_dir = os.path.join(os.getcwd(), "data")
    return os.path.join(data_dir, f"{filename}.xlsx")


def load_news_contents():
    file_path = get_full_path("crawling_dataset")

    news_df = read_excel(
        file_path,
        sheet_name="News",
        index_col=0,
    )

    return news_df["news_content"].tolist()


def load_comment_contents():
    file_path = get_full_path("crawling_dataset")

    comment_df = read_excel(
        file_path,
        sheet_name="Comments",
        index_col=[0, 1],
    )

    return comment_df["comment_content"].tolist()
