# ORE

<br>

[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)


## Intorduction:
Inspired by the amazing effect of [Magi](https://magi.com/), we try to approach a more simple, brutal method to handle the open relation extraction problem, only using a pipeline including NER and RE.

<br>

## Definition:
**Open Relation Extraction**, also defined as **Open Triples Extraction** or **Open Facts Extraction**, is a task to extract any triples (head entity, tail entity, relation) / tuples (head entity, tail entity) from the given text (a sentence or a document, we focus on sentence here). Our method **ORE** is short for Open Relation Extraction, utilized the pre-trained language model **BERT** as backbone, implemented by **PyTorch**.

<br>

## Pipeline
There are three main steps of the pipeline:
**DS** -> **OpenNER** -> **OpenRE**

* **DS: Distant Supervision**: split each long text into sentences. Located triples (sometimes named facts) from knowledge graph in each sentence. That is, one sample has one sentence, several triples (if relation is not None), tuples (the current entity pair has relation but the relation span does not exist in the sentence), and some isolated entities. Entity linking, correference resolution, and the modifier-core structure expansion might be necessary.

* **OpenNER: Open Named Entity Recognition**: train a NER model to find all entities in the current sentence, regardless of the type of entities. B-I-O annotation strategy is used here.

* **OpenRE: Open Relation Extraction**: do the piecewise coupling for all entities recognized by NER model, and input each entity-pair with the sentence into a RE model, which predicts whether the two entities have relation or not. If there is a relation existing between the current two entities, RE model would further tried to find the relation span.

* **Note** We do not provide detailed approaches of Distant Supervision.

<br>

## File Dependency:
```
-> your_raw_data -> your_train_data.json
                 |-> your_test_data.json
-> check_points -> bert_chinese_ner -> log.txt
                                    |-> model.pth
                                    |-> setting.txt
                -> bert_chinese_re -> log.txt
                                   |-> model.pth
                                   |-> setting.txt
pretrain_data -> ner_train.pkl
              |-> ner_dev.pkl
              |-> re_train.pkl
              |-> re_dev.pkl
pretrain_models -> bert_chinese -> bert_config.json
                                |-> pytorch_model.pth
                                |-> vocab.txt
bert_codes -> python scripts of tokenization, modeling, optimization, utils
learn_ner.py
learn_re.py
```

<br>

## Dataset
* **raw corpus**: The raw corpus we used for this work is from [Magi Practical Web Article Corpus](https://zenodo.org/record/3242512#.XuwvMBMza2w). Factually, any unstructured texts are suitable. <br>

* **knowledge graphs**: The main knowledge graph we used for distant supervision is  [CN-DBpedia](http://kw.fudan.edu.cn/cndbpedia/download/). Fusing several graphs may be useful.

* **training data**: examples could be found [here](https://github.com/Schlampig/ORE/blob/master/data_examples/toy_train_data.json). The type of both head_entity, tail_entity, and relation is string, while sometimes relation could be None. In addition, head/tail_entity is a string denoted as "entity mention" in each text, entity_index is the index of head/tail_entity. If there are more than one head/tail_entity in one text, entity_index = [idx_1, idx_2, …]. Whatever, entity_index is actually not used in this method.
```
sample = {"_id": string, 
          "EL_res": [{"text": string, 
                      "triples": [[head_entity, tail_entity, relation], 
                                   [head_entity, tail_entity, relation],
                                   ...], 
                      "entity_idx": {head/tail_entity: entity_index, ...}
                      }, 
                     {"text": string, 
                      "triples": list, 
                      "entity_idx": dictionary},
                      ...
                      ]}
```
* **test data**: examples could be found [here](https://github.com/Schlampig/ORE/blob/master/data_examples/toy_test_data.json). The type of both head_entity, tail_entity, and relation is string, while sometimes relation could be None.
```
test_data = [
            {"unique_id": int, 
             "text": string, 
             "triples": [[head_entity, tail_entity, relation],
                         [head_entity, tail_entity, relation],
                         ...]
            },
            {"unique_id": int, 
             "text": string, 
             "triples": list},
             ...
]
```

<br>

## Command Line:
* **learn NER model**: you could prepare data, train the model, make one prediction for the given example all together by just running the following command (or you could go into the learn_ner.py script to run them in turn):
```bash
python learn_ner.py
```
* **learn RE model**: you could prepare data, train the model, make one prediction for the given example, make prediciton on batched test data all together by just running the following command (or you could go into the learn_re.py script to run them in turn):
```bash
python learn_re.py
```

<br>

## Requirements
  * Python = 3.6.9
  * pytorch = 1.3.1
  * scikit-learn = 0.21.3
  * tqdm = 4.39.0
  * requests = 2.22.0 (optional)
  * Flask = 1.1.1 (optional)
  * ipdb = 0.12.2 (optional)

<br>

## References
* **code**: the original BERT-related codes are from [bert_cn_finetune](https://github.com/ewrfcas/bert_cn_finetune) project of [ewrfcas](https://github.com/ewrfcas) and [transformers](https://github.com/huggingface/transformers) project of [Hugging Face](https://github.com/huggingface). <br>
* **literature**: 
  - Soares, L. B. , FitzGerald, N. , Ling, J. , Kwiatkowski T. . (2019). Matching the Blanks: Distributional Similarity for Relation Learning. ACL 2019. [paper](https://arxiv.org/abs/1906.03158?context=cs.CL)/[code](https://github.com/zhpmatrix/BERTem).
  - Zhang, N. , Deng, S. , Sun, Z. , Wang, G. , Chen, X. , Zhang, W. , Chen, H. . (2019). Long-tail Relation Extraction via Knowledge Graph Embeddings and Graph Convolution Networks. NAACL 2019. [paper](https://arxiv.org/abs/1903.01306v1).
  - Jianlin Su. A Hierarchical Relation Extraction Model with Pointer-Tagging Hybrid Structure. GitHub. [blog](https://kexue.fm/archives/6671)/[code](https://github.com/bojone/kg-2019).
  - Gabriel Stanovsky, Julian Michael, Luke Zettlemoyer, Ido Dagan. Supervised Open Information Extraction. NAACL 2018. [paper](https://www.researchgate.net/publication/325445833_Supervised_Open_Information_Extraction).

<br>

<img src="https://github.com/Schlampig/Knowledge_Graph_Wander/blob/master/content/daily_ai_paper_view.png" height=25% width=25% />
