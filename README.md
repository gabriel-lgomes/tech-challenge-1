# Tech Challenge Fase 1 — IA para Saúde Feminina

Sistema inteligente de suporte ao diagnóstico médico com foco na saúde e segurança da mulher, utilizando Machine Learning para identificação precoce de condições de risco.

## Objetivo

Construir modelos preditivos de classificação sobre dados médicos relacionados à saúde feminina, aplicando técnicas de Machine Learning, análise exploratória e explicabilidade dos resultados.

## Dataset

**Breast Cancer Wisconsin Diagnostic Dataset**  
Classificação de tumores de mama como malignos ou benignos a partir de características extraídas de imagens de biópsia.

- Fonte: [Kaggle — UCI ML Breast Cancer Wisconsin](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
- 569 amostras | 30 features | Variável alvo: `diagnosis` (M = maligno, B = benigno)

## Estrutura do projeto

```
tech-challenge-1/
├── data/
│   └── processed/       # Dados após pré-processamento
├── notebooks/
│   ├── 01_eda.ipynb              # Análise exploratória
│   ├── 02_preprocessing.ipynb    # Pré-processamento e pipeline
│   └── 03_modeling.ipynb         # Modelagem, avaliação e SHAP
├── src/
│   ├── preprocessing.py  # Pipeline de pré-processamento
│   └── evaluation.py     # Funções de avaliação e métricas
├── models/               # Modelos treinados (não versionados)
├── reports/
│   └── figures/          # Gráficos e visualizações geradas
├── requirements.txt
└── README.md
```

## Modelos utilizados

- Regressão Logística
- Random Forest
- (+ comparativo de performance entre os dois)

## Métricas de avaliação

Dado o contexto médico, o **Recall** é a métrica mais crítica — um falso negativo (não detectar um tumor maligno) tem consequências graves para a paciente. São reportadas também accuracy, precision e F1-score.

## Explicabilidade

- Feature importance (Random Forest)
- SHAP values para interpretação global e individual das predições

## Como executar

### Pré-requisitos

- Python 3.10+
- pip

### Instalação

```bash
pip install -r requirements.txt
```

### Execução

Execute os notebooks na ordem numérica dentro da pasta `notebooks/`:

1. `01_eda.ipynb` — Análise exploratória dos dados
2. `02_preprocessing.ipynb` — Limpeza, encoding e pipeline de features
3. `03_modeling.ipynb` — Treinamento, avaliação e explicabilidade

O dataset deve ser baixado do Kaggle e colocado em `data/raw/` antes de executar.

## Resultados

> Seção a ser preenchida após a execução dos modelos.

## Discussão crítica

Os modelos desenvolvidos têm caráter de **apoio à decisão**, não de diagnóstico autônomo. O médico responsável deve sempre ter a palavra final, utilizando as predições como uma ferramenta auxiliar de triagem.
