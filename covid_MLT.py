from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import pandas as pd

from data.covid_data_process import (
    parse_data,
    pu_label_process_trans,
    read_ris_file,
)

from utils import (
    get_metric,
    set_seed,
)

es = Elasticsearch("http://127.0.0.1:9200", verify_elasticsearch=False)
health = es.cluster.health()
print("cluster health:", health)

data_dir = r"/root/autodl-tmp/PU_all_in_one/data/Cochrane_Covid-19"
settings_mode = 3
num_lp = 50
random_state = 42
set_seed(random_state)
TrainingIncludes = read_ris_file(data_dir, "1_Training_Included_20878.ris.txt")
TrainingExcludes = read_ris_file(data_dir, "2_Training_Excluded_38635.ris.txt")
CalibrationIncludes = read_ris_file(data_dir, "3_Calibration_Included_6005.ris.txt")
CalibrationExcludes = read_ris_file(data_dir, "4_Calibration_Excluded_10118.ris.txt")
EvaluationIncludes = read_ris_file(data_dir, "5_Evaluation_Included_2310.ris.txt")
EvaluationExcludes = read_ris_file(data_dir, "6_Evaluation_Excluded_2412.ris.txt")

if settings_mode == 1:
    CalibrationDf = parse_data(TrainingIncludes, TrainingExcludes)
    TrainingDf = parse_data(CalibrationIncludes, CalibrationExcludes)
    EvaluationDf = parse_data(EvaluationIncludes, EvaluationExcludes)
elif settings_mode == 2:
    EvaluationDf = parse_data(TrainingIncludes, TrainingExcludes)
    CalibrationDf = parse_data(CalibrationIncludes, CalibrationExcludes)
    TrainingDf = parse_data(EvaluationIncludes, EvaluationExcludes)
elif settings_mode == 3:
    TrainingDf = parse_data(TrainingIncludes, TrainingExcludes)
    CalibrationDf = parse_data(CalibrationIncludes, CalibrationExcludes)
    EvaluationDf = parse_data(EvaluationIncludes, EvaluationExcludes)
else:
    TrainingDf, CalibrationDf, EvaluationDf = None, None, None
    print("Invalid settings mode.")

tr_df, _, _ = pu_label_process_trans(
    TrainingDf, CalibrationDf, EvaluationDf, num_lp, random_state
)

train_df = tr_df.query("tr == 1" and "pulabel==1")
test_df = tr_df.query("ts == 1")
if es.indices.exists(index="training_articles"):
    es.indices.delete(index="training_articles")
es.indices.create(index="training_articles")


def generate_data(df):
    for index, row in df.iterrows():
        yield {
            "_index": "training_articles",
            "_id": index,
            "_source": {
                "title": row["title"],
                "abstract": row["abstract"],
            },
        }


bulk(es, generate_data(train_df))


def score_test_articles(test):
    scores = []
    for index, row in test.iterrows():
        query = {
            "more_like_this": {
                "fields": ["title", "abstract"],
                "like": [{"doc": {"title": row["title"], "abstract": row["abstract"]}}],
                "min_term_freq": 1,
                "max_query_terms": 25,
            }
        }
        results = es.search(index="training_articles", body={"query": query}, size=1)
        top_score = (
            results["hits"]["hits"][0]["_score"] if results["hits"]["hits"] else 0
        )
        scores.append((index, top_score))

    return scores


test_scores = score_test_articles(test_df)
test_scores = pd.DataFrame(test_scores, columns=["index", "score"])
test_labels = tr_df.query("ts == 1")["label"].values
MLT_test_info_tuple = get_metric(test_labels, test_scores.score)
metrics = [
    "threshold",
    "threshold99",
    "auc",
    "f1",
    "acc",
    "rec",
    "f1_99",
    "acc_99",
    "rec_99",
    "r_10",
    "r_20",
    "r_30",
    "r_40",
    "r_50",
    "r_95",
    "reduce_work",
    "p_mean",
    "n_mean",
    "WSS95",
    "WSS100",
    "p_LastRel",
    "prec10",
    "prec20",
    "recall10",
    "recall20",
]
print(metrics, MLT_test_info_tuple)
