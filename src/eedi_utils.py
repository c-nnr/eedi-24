import polars as pl

import os
from dataclasses import dataclass

@dataclass
class Config:
    validation_set: str
    comp_data_dir: str = 'data/eedi-24'
    is_submission: bool = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))

def get_inference_dataframe(config: Config) -> pl.DataFrame:
    df = pl.read_csv(f'{config.comp_data_dir}/{config.validation_set}.csv')
    answer_df = df.unpivot(
        index=["QuestionId", "ConstructName", "QuestionText", "CorrectAnswer"],
        on=[f"Answer{c}Text" for c in ["A", "B", "C", "D"]],
        variable_name = "Answer",
        value_name = "AnswerText",
    ).with_columns(
        (pl.col("QuestionId").cast(pl.String) + "_" + pl.col("Answer").str.slice(6, 1)).alias("QuestionId_Answer"),
        (pl.col("QuestionId").cast(pl.String) + "_" + pl.col("CorrectAnswer")).alias("CorrectAnswerId"),
        (pl.col("CorrectAnswer") == pl.col("Answer").str.slice(6, 1)).alias('IsCorrect')
    )
    correct_answers = answer_df.filter(pl.col("IsCorrect")).drop("CorrectAnswerId").rename({"QuestionId_Answer": "CorrectAnswerId"}).select("CorrectAnswerId", "AnswerText")
    tidy_df = answer_df.filter(~pl.col("IsCorrect"))
    tidy_df = tidy_df.join(correct_answers, on="CorrectAnswerId", how='left').select("QuestionId_Answer",  "QuestionId", "ConstructName", "QuestionText", "AnswerText", "AnswerText_right").rename({"AnswerText_right": "CorrectAnswerText"}).sort(by="QuestionId")
    tidy_df = tidy_df.with_columns(
        ("### Question\n"+pl.col("QuestionText")+"\n\n### Context\nKnowleged Tested: "+pl.col("ConstructName")+"\nCorrect Answer: "+pl.col("CorrectAnswerText")+"\n\n### Given Answer\n"+pl.col("AnswerText")).alias("PromptText")
    ).select("QuestionId_Answer", "PromptText")
    return tidy_df

def get_complete_dataframe(config: Config) -> pl.DataFrame:
    mismap_df = pl.read_csv(f'{config.comp_data_dir}/misconception_mapping.csv').with_columns(pl.col("MisconceptionId").cast(pl.Float64))
    df = pl.read_csv(f'{config.comp_data_dir}/train.csv')
    if config.validation_set=='test':
        df = df.head(n=3)
        df = df.with_columns(pl.col("QuestionId") + 1869)
    miscon_df = df.unpivot(
        index=["QuestionId"],
        on=[f"Misconception{c}Id" for c in ["A", "B", "C", "D"]],
        variable_name = "Misconception",
        value_name = "MisconceptionId",
    ).with_columns(
        (pl.col("QuestionId").cast(pl.String) + "_" + pl.col("Misconception").str.slice(13, 1)).alias("QuestionId_Answer"),
    ).select("QuestionId_Answer", "MisconceptionId")
    miscon_df = miscon_df.join(mismap_df, on="MisconceptionId", how="left")
    tidy_df = get_inference_dataframe(config)
    tidy_df = tidy_df.join(miscon_df, on="QuestionId_Answer", how='left').drop_nulls('MisconceptionId')
    return tidy_df 

def score(config: Config, preds: pl.DataFrame) -> float:
    gt = get_complete_dataframe(config).rename({"MisconceptionId":"GT_MisconceptionId"})
    scoring = gt.join(preds, on="QuestionId_Answer", how="left").with_columns(pl.col("MisconceptionId").str.split(" ").list.contains(pl.col("GT_MisconceptionId").cast(pl.Int64).cast(pl.String)).alias("correct@25"))
    return scoring.select(pl.col("correct@25").mean()).item()