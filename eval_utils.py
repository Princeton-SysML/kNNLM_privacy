import json
import pathlib
import re
from multiprocessing import Pool, Queue
from typing import Dict

import numpy as np
import pandas as pd

# import spacy
import torch
import tqdm

# from rouge_score import rouge_scorer

import spacy

NER = spacy.load("en_core_web_sm")

# def namedEntityRatio(nes_ori, nes_cand):
#     nes_ori = set(nes_ori)

#     if len(nes_ori) == 0:
#         return 0
#     if len(nes_cand) == 0:
#         return 0

#     nes_cand = set(nes_cand)

#     recall = len(nes_ori.intersection(nes_cand)) / len(nes_ori)
#     precision = len(nes_ori.intersection(nes_cand)) / len(nes_cand)
#     if recall + precision == 0:
#         return 0
#     return 2 * (recall * precision) / (recall + precision)


# def nerScore(original, generated, nlp):
#     cost = np.zeros((len(original), len(generated)))
#     named_entities_original = [
#         list(map(lambda x: x.text, nlp(o).ents)) for o in original
#     ]
#     named_entities_candidate = [
#         list(map(lambda x: x.text, nlp(o).ents)) for o in generated
#     ]
#     for i, o in enumerate(named_entities_original):
#         for j, g in enumerate(named_entities_candidate):
#             cost[i, j] = namedEntityRatio(o, g)

#     return cost, named_entities_original, named_entities_candidate


# def rougeScores(original, generated, scorer):
#     rouge_scores = {}
#     for statistic in ["fmeasure", "recall", "precision"]:
#         cost_1 = np.zeros((len(original), len(generated)))
#         cost_2 = np.zeros((len(original), len(generated)))
#         cost_L = np.zeros((len(original), len(generated)))
#         for i, o in enumerate(original):
#             for j, g in enumerate(generated):
#                 scores = scorer.score(o, g)
#                 if statistic == "fmeasure":
#                     cost_1[i, j] = scores["rouge1"].fmeasure
#                     cost_2[i, j] = scores["rouge2"].fmeasure
#                     cost_L[i, j] = scores["rougeL"].fmeasure
#                 elif statistic == "recall":
#                     cost_1[i, j] = scores["rouge1"].recall
#                     cost_2[i, j] = scores["rouge2"].recall
#                     cost_L[i, j] = scores["rougeL"].recall
#                 elif statistic == "precision":
#                     cost_1[i, j] = scores["rouge1"].precision
#                     cost_2[i, j] = scores["rouge2"].precision
#                     cost_L[i, j] = scores["rougeL"].precision
#         rouge_scores[statistic] = {"rouge1": cost_1, "rouge2": cost_2, "rougeL": cost_L}

#     return rouge_scores


# def createScorers():
#     nlp = spacy.load("en_core_web_sm")
#     rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
#     return {"nlp": nlp, "rouge": rouge}


# def calcMetrics(ref, cand, scorers, strip=True):
#     if strip:
#         cand = ["".join(re.split(r'( *[\.\?!][\'"\)\]]* *)', c)[:2]) for c in cand]
#     ner, ent_orig, ent_cand = nerScore(ref, cand, scorers["nlp"])
#     df = pd.DataFrame(
#         {
#             "original": [ref],
#             "candidate": [cand],
#             "ner": [ner],
#             "ent_orig": [ent_orig],
#             "ent_cand": [ent_cand],
#         }
#     )

#     rouge_scores = rougeScores(ref, cand, scorers["rouge"])
#     for statistic in ["fmeasure", "recall", "precision"]:
#         for rouge in ["1", "2", "L"]:
#             df[f"rouge_{rouge}_{statistic}"] = [
#                 rouge_scores[statistic][f"rouge{rouge}"]
#             ]

#     return df


# def getTopkMetrics(result, params, scorers, k=1):
#     topk_data = [
#         calcMetrics(result["original"], result["results"][i]["result"], scorers)
#         for i in range(0, k)
#         if len(result["results"]) > i
#     ]
#     for idx, metrics in enumerate(topk_data):
#         for (k, v) in params.items():
#             metrics[k] = v
#         metrics["topk"] = idx
#     topk_data = pd.concat(topk_data, ignore_index=True)
#     return topk_data


# def getAllResultsJson(dir: pathlib.Path):
#     for child in dir.iterdir():
#         if child.name == "result.json":
#             params = child.parent / "params.json"
#             params = json.load(params.open())
#             results = json.load(child.open())
#             if params["no_repeat_ngram_size"] is None:
#                 params["no_repeat_ngram_size"] = 0
#             yield {"results": results, "params": params}
#         elif child.is_dir():
#             yield from getAllResultsJson(child)


# def isNewRow(dataframe: pd.DataFrame, parameters: Dict):
#     dataframe_columns = set(dataframe.columns)
#     for (k, v) in parameters.items():
#         if k not in dataframe_columns:
#             return True
#         dataframe = dataframe[dataframe[k] == v]
#     return len(dataframe) == 0


# def calculateAllNewMetrics(
#     directory: pathlib.Path, existing: pd.DataFrame = None, topk=1
# ):
#     all_results = getAllResultsJson(directory)
#     if existing is not None:
#         all_results = filter(lambda r: isNewRow(existing, r["params"]), all_results)
#     else:
#         existing = pd.DataFrame()

#     scorers = createScorers()
#     dataframes = []
#     for result in tqdm.tqdm(list(all_results)):
#         dataframes.append(
#             getTopkMetrics(result["results"], result["params"], scorers, topk)
#         )
#     return dataframes


def calculatePerplexity(sentence, model, tokenizer):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return torch.exp(loss)


def count_max_same_seq(sent):
    token_list = sent.split(" ")
    max_same = 1
    tmp_max = 1
    last_t = token_list[0]
    for t in token_list[1:]:
        if t == last_t:
            tmp_max += 1
            if tmp_max > max_same:
                max_same = tmp_max
        else:
            if tmp_max > max_same:
                max_same = tmp_max
            tmp_max = 1
        last_t = t
    return max_same


def deduplicate(res_f, repetition_penalty=0.75, no_repeat_ngram_size=0):
    res = pd.read_csv(
        res_f,
        on_bad_lines="skip",
        header=None,
        index_col=None,
        sep=";;",
        engine="python",
    )
    res.columns = [
        "prompt_idx",
        "generate_id",
        "sequence",
        "beam score",
        "PPL",
        "ref PPL",
    ]

    res_dedup = res.drop_duplicates(subset=["sequence"])
    res_dedup.reset_index(inplace=True)

    # 1. Remove sentences with > 10 consecutive same tokens
    keep_indices = []
    stats_list = []
    for i in range(len(res_dedup)):
        sent = res_dedup["sequence"][i]
        max_rep_len = count_max_same_seq(sent)
        if max_rep_len < 4:
            keep_indices.append(i)
        stats_list.append(max_rep_len)

    res_cleaned = res.iloc[keep_indices]
    print(
        f"repetition_penalty: {repetition_penalty}, no_repeat_ngram_size: {no_repeat_ngram_size}    Original: {len(res)} sentences  After de-duplication: {len(res_dedup)} sentences After de-repetition: {len(res_cleaned)} sentences"
    )
    return res_cleaned


def run_eval_sens(
    recon_candidate, subsets=["person", "location", "date", "phone", "email", "url"]
):
    patterns = {
        "url": r'https?://[^\s<>"]+|www\.[^\s<>"]+',
        "phone": r"[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}",
        "email": r"[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+",
    }
    ents = {
        "date": ["DATE"],
        "location": ["LOC", "GPE"],
        "person": "PERSON",
    }

    res_stats = {}
    matched_all = {}
    res_df = pd.DataFrame()
    res_df["sent"] = recon_candidate

    recon_ents = [
        d
        for d in NER.pipe(
            recon_candidate,
            disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"],
        )
    ]
    recon_ents_list = {}
    recon_ents_list["person"] = [
        [ent.text for ent in doc.ents if ent.label_ in ents["person"]]
        for doc in recon_ents
    ]
    recon_ents_list["date"] = [
        [ent.text for ent in doc.ents if ent.label_ in ents["date"]]
        for doc in recon_ents
    ]
    recon_ents_list["location"] = [
        [ent.text for ent in doc.ents if ent.label_ in ents["location"]]
        for doc in recon_ents
    ]

    for subset in subsets:
        print(subset)
        targets = pd.read_csv(f"data/enron/sensitive/{subset}.csv")
        targets_items = [ast.literal_eval(t) for t in targets["item"].tolist()]
        targets_items = list(set(list(chain(*[t for t in targets_items]))))

        recon_candidate_items = [None] * len(recon_candidate)

        for idx, recon in enumerate(recon_candidate):
            if subset in ["url", "email", "phone"]:
                pattern = patterns[subset]
                item_recons = re.findall(pattern, recon)
            if subset in ["date", "location", "person"]:
                item_recons = recon_ents_list[subset][idx]
            if len(item_recons) > 0:
                recon_candidate_items[idx] = item_recons
            else:
                recon_candidate_items[idx] = []

        res_df[f"recon_{subset}"] = recon_candidate_items

        recon_all = []
        for m in res_df[f"recon_{subset}"].values:
            recon_all.extend(m)

        matched_all[subset] = set(recon_all).intersection(set(targets_items))
        res_stats[f"num_unique_{subset}"] = len(matched_all[subset]) - int(
            "" in matched_all[subset]
        )
    return res_stats, res_df, matched_all
