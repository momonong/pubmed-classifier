# KG-GNN-pubmed

```
poetry run pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

```
poetry run python -m spacy info en_core_sci_sm
```

| 分類                    | 對應階段           | 說明                                                             |
| --------------------- | -------------- | -------------------------------------------------------------- |
| **T0 / T1** → Label 0 | 初步轉譯階段（T1）     | 著重於基礎研究應用（如新基因的發現、測試開發等）——也就是「bench-to-bedside」階段              |
| **T2–T4** → Label 1   | 後期轉譯階段（T2\~T4） | 涵蓋 **臨床效益驗證、實務應用推廣與真實世界健康成效評估**，也就是「beyond bench-to-bedside」研究 |
