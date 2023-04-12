from torch.utils.data import Dataset
from .ReadNewsDataset import load_comment_contents


class CommentDataset(Dataset):
    def __init__(self):
        df_comment = load_comment_contents()
        self.content = df_comment["content"].tolist()
        self.label = df_comment["label"].tolist()

    def __getitem__(self, index):
        return self.content[index], self.label[index]

    def __len__(self):
        return len(self.content)
