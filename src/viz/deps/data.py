import pandas as pd
from src.viz.deps.types import PaperFactory


class PaperData(pd.DataFrame):
    def __init__(self, path: str):
        super().__init__(
            data=pd.DataFrame.from_dict(
                PaperFactory.many_from_json(path),
                orient='index'
            )
        )


if __name__ == '__main__':
    df = PaperData('../data/sample.json')
    print(df.to_string())
