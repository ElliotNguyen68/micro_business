from typing import List
import pandas as pd


def join(left_df:pd.DataFrame,right_df:pd.DataFrame,on:List[str],how:str)->pd.DataFrame:
    return left_df.merge(right=right_df,on=on,how=how)