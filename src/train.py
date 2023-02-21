from datetime import datetime,date
from dateutil.relativedelta import relativedelta

from tsfresh.feature_extraction import extract_features
import pandas as pd


def create_feature_ts_for_dt(
    df_continuos:pd.DataFrame,
    dt:datetime,
    df_feature_active:pd.DataFrame,
    target_col:str,
    mcb_col:str,
    time_col:str
)->pd.DataFrame:

    df_continuos_1=df_continuos.copy(deep=True).rename({target_col:'target_1',time_col:'time_1'},axis=1).drop([mcb_col,'row_id'],axis=1)
    df_continuos_2=(
        df_continuos.loc[lambda x:x.first_day_of_month==(dt-relativedelta(months=1))]
        .copy(deep=True).rename({target_col:'target_2',time_col:'time_2'},axis=1).drop([mcb_col],axis=1)
    )

    df_join=pd.merge(
        df_continuos_2,
        df_continuos_1,
        on='cfips',
    ).loc[lambda x:x.time_2>=x.time_1][['row_id','target_1','time_1']].sort_values(['row_id','time_1'])

    df_ts=extract_features(df_join,column_id='row_id',column_sort='time_1',n_jobs=32).reset_index(names=['row_id'])

    df_ts=(
        df_ts.assign(cfips=lambda x:x.row_id.str.split('_').str[0].apply(int))
        .assign(first_day_of_month=lambda x:x.row_id.str.split('_').str[1])
        .assign(first_day_of_month=lambda x:pd.to_datetime(x.first_day_of_month))
    )

    df_feature_active=(
        df_feature_active.copy(deep=True)
        .assign(cfips=lambda x:x.row_id.str.split('_').str[0].apply(int))
    )
    df_tmp=pd.merge(
        df_ts,
        df_feature_active.drop(['row_id'],axis=1).groupby('cfips').last().reset_index(),
        on='cfips'
    )


    return df_tmp

