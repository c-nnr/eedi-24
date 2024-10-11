import polars as pl

import os
from dataclasses import dataclass

@dataclass
class Config:
    model_id : str
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

def few_shot_prompt():
    return """\
### Question
Expand and simplify:
\[
4(2 x+1)-(5 x-9)
\]

### Context
Knowleged Tested: Multiply a single term over a bracket where the term on the outside is a number and the inside contains a linear expression
Correct Answer: \( 3 x+13 \)

### Given Answer
\( 13 x-5 \)

### Misconception
Adds instead of subtracts

### Question
\( (-9)+(-4)= \)

### Context
Knowleged Tested: Carry out addition problems involving two negative integers 
Correct Answer: \( -13 \)

### Given Answer
\( 13 \)

### Misconception
Believes adding two negatives gives a positive answer

### Question
Which symbol makes the following statement correct?
\(
\frac{3}{4}\space \square\space \frac{5}{8}
\)

### Context
Knowleged Tested: Use inequality notation to order fractions whose denominators are multiples of the same number 
Correct Answer: \( > \)

### Given Answer
\( = \)

### Misconception
Believes that fractions with larger denominators are greater in size

### Question
Tom and Katie are discussing graphs that show the volume of water
(V) flowing from a kitchen tap over time (t). Tom says this graph shows water flowing at a constant rate ![A sketch of a graph showing V on the y axis and t on the x axis. A line starts at the origin and slopes upwards getting gradually steeper. ]() Katie says this graph shows water flowing at a constant rate ![A sketch of a graph showing V on the y axis and t on the x axis. A horizontal straight line goes from halfway up the y axis straight across the graph. ]() Who is correct?

### Context
Knowleged Tested: Compare real life graphs
Correct Answer: Only Katie

### Given Answer
Only Tom

### Misconception
Believes a curve can show a constant rate

### Question
Which of the following is the square root of \( 36 \) ?

### Context
Knowleged Tested: Recognise square roots
Correct Answer: \( 6 \)

### Given Answer
\( 72 \)

### Misconception
Mixes up square rooting and multiplying by 2 or doubling

### Question
Here is a number line: ![A horizontal number-line with 13 vertical, equally spaced, dashes representing the position of numbers. The 1st dash is labelled with "-6", the 3rd dash is labelled "-4", the 5th dash is labelled "-2", the 7th dash is labelled "0", the 9th dash is labelled "2", the 11th dash is labelled "4" and the 13th dash is labelled "6"    A red arrow, labelled with a question mark is pointing halfway between the 6th and 7th dashes. ]() Which of the following numbers could be indicated by the arrow?

### Context
Knowleged Tested: Identify where negative non-integers lie on a number line
Correct Answer: \( -0.5 \)

### Given Answer
\( -1.5 \)

### Misconception
Counts on in the wrong direction on a number line

### Question
According to the graph, what is the approximate solution to
\(x^{2}-3 x-6=2\) ![A set of axes with the quadratic graph y=x^2-3x-6 drawn on.]()

### Context
Knowleged Tested: Given a sketch of a quadratic graph, f(x), write down the solutions to f(x) = a
Correct Answer: \( x=-1.75, \quad x=4.75 \)

### Given Answer
\( y=-6 \)

### Misconception
Finds the y intercept when asked to read the solution to a quadratic equation from a graph

### Question
What should replace the circle when these two brackets are expanded and simplified?
\(
(p+5)(p-4) \equiv p^{2} \triangle p \bigcirc
\)

### Context
Knowleged Tested: Expand two brackets with linear terms in the form (x + a)(x + b)
Correct Answer: \( \bigcirc=-20 \)

### Given Answer
\( \bigcirc=+1 \)

### Misconception
Adds instead of multiplying when expanding bracket

### Question
What do the dashes tell you about the sides of this polygon? ![A hexagon with a dash on each side.]()

### Context
Knowleged Tested: Understand the terms equilateral and regular, and recognise the notation for regular polygons
Correct Answer: They are the same length

### Given Answer
They are parallel

### Misconception
Confuses the sign for parallel lines as being dashes

### Question
What is the correct name for the line marked on the circle? ![A circle with a line from the centre of the circle to the circumference.]()

### Context
Knowleged Tested: Identify a radius
Correct Answer: Radius

### Given Answer
Diameter

### Misconception
Confuses diameter and radius

"""