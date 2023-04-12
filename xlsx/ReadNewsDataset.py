import os
from pandas import read_excel


def get_full_path(filename: str):
    data_dir = os.path.join(os.getcwd(), "data")
    return os.path.join(data_dir, f"{filename}.xlsx")


def load_news_contents():
    file_path = get_full_path("crawling_data")

    return read_excel(
        file_path,
        sheet_name="News",
        index_col=0,
    )


def load_comment_contents():
    file_path = get_full_path("crawling_data")

    return read_excel(
        file_path,
        sheet_name="Comments",
        index_col=[0, 1],
    )
