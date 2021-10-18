# Relation_Extraction
문장 안의 entity 사이의 관계 추출의 성능을 높일 수 있는 방법에 대해 연구한 프로젝트입니다. (2021.03 ~ 2021.05)

## Project Goal
Knowledge Graph는 많은 AI task(semantic search, question answering, recommendation)에 활용되어 성능을 높이는데 기여할 수 있다는 연구 결과가 있습니다.

이처럼 Knowledge Graph는 활용 가능성이 많지만, 구축하는데 인건비 등의 많은 비용이 필요합니다.

이러한 비용적인 측면의 비효율성 때문에 최근에는 NLP를 이용해 Knowledge Graph를 기계적으로 구축하려는 많은 연구가 이루어지고 있습니다.

이 프로젝트에서는 기계적인 Knowledge Graph Construction의 핵심이라 할 수 있는 **Relation Extraction Task**와 관련된 연구를 하고자 합니다.

좀 더 구체적으로는 **BERT**를 이용해 Relation Extraction Task의 성능 개선과 성능에 영향을 미치는 요인에 대해 연구하는 것이 목적입니다.

## Ideas
- BERT 통과 후 representation vector 변형하여 학습
- SBERT의 multilingual knowledge distillation 이용(Teacher-Student model)
- SBERT 이용
- Triplet Learning 이용한 pretrain
- Triplet Learning의 anchor 설정 기준에 따른 실험

## Results
문장의 구조적인 정보(entity distance, sentence length)와 Relation Extraction의 성능 사이의 관계에 대해 연구를 진행하였습니다.

Triplet Learning 방식을 차용하여 Relation Extraction에 적용하였고, 이를 통해 성능을 개선시킬 수 있음을 보여주었습니다.

BERT 기반의 학습을 통해 Relation Extraction을 할 때 문장의 구조적인 정보가 성능에 큰 영향을 미치지는 못한다는 결론을 내렸지만, Triplet Learning 방식을 사용하여 Relation Extraction을 했다는 점과, 이 때 anchor는 random sampling하는 방식이 성능을 가장 향상시킬 수 있다는 것을 보여주는데 의의가 있는 연구입니다.

## Data
Pretrain용 데이터와 성능 평가 데이터셋은 [RE-Context-or-Names](https://github.com/thunlp/RE-Context-or-Names)에서 다운로드받으실 수 있습니다.

## GPU
Pretrain에 사용한 GPU 수는 4대이고, finetuning에 사용한 GPU 수는 1대입니다.