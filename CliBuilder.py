import pandas as pd
from dataclasses import dataclass
from functools import reduce
from typing import Callable, Dict, List, Tuple

class CliBuilder():
    def __init__(self, df_re, df_ent, schema: Schema = SCHEMA):
        self.schema = schema
        self.df_re = df_re
        self.df_ent = df_ent

    def _attach_mt(self, frame: pd.DataFrame, side: str) -> pd.DataFrame:
        s = self.schema
        type_col, id_col = s.key_cols(side)
        merged = pd.merge(
            frame, self.df_ent,
            left_on=[type_col, id_col],
            right_on=[s.ent_type, s.ent_id],
            how='inner',
        ).drop(s.helper_cols, axis=1)
        return merged.rename(columns={s.mt: f"{side}_{s.mt}"})

    def search_ent_mt(self):
        steps: List[Callable[[pd.DataFrame], pd.DataFrame]] = [
            (lambda frame, side=side: self._attach_mt(frame, side))
            for side in self.schema.sides
        ]
        return reduce(lambda acc, step: step(acc), steps, self.df_re)


    def _split_composite(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        s = self.schema
        parts = series.str.split(s.id_delim, expand=True)
        return parts[0], parts[1].str[s.id_prefix_len:]

    def _validate(self, df: pd.DataFrame, required: List[str]) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

    def REclib_to_df(self, df):
        s = self.schema
        self._validate(df, list(s.passthrough) + [s.id_col_name(x) if False else x + s.id_suffix for x in s.sides])

        parsed: Dict[str, pd.Series] = {c: df[c].reset_index(drop=True) for c in s.passthrough}
        for side in s.sides:
            type_col, id_col = s.key_cols(side)
            t, i = self._split_composite(df[id_col].reset_index(drop=True))
            parsed[type_col], parsed[id_col] = t, i

        ordered = list(s.passthrough) + [c for side in s.sides for c in s.key_cols(side)]
        return pd.DataFrame({c: parsed[c] for c in ordered}, columns=ordered)

    def ent_to_df(self, df):
        s = self.schema
        df[s.ent_id] = df[s.ent_id].str.split(s.put_delim).str[1]
        return df