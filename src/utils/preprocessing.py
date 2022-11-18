import pandas as pd

def label_encode(col: pd.core.series.Series) -> pd.core.series.Series:
    return col.astype('category').cat.codes
