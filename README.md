# Tech Challenge Fase 1 — IA para Saúde Feminina

Sistema inteligente de suporte ao diagnóstico médico com foco na saúde e segurança da mulher, utilizando Machine Learning para identificação precoce de câncer de mama.

## Objetivo

Construir modelos preditivos de classificação sobre dados médicos relacionados à saúde feminina, aplicando técnicas de Machine Learning, análise exploratória e explicabilidade dos resultados.

## Dataset

**Breast Cancer Wisconsin Diagnostic Dataset**
Classificação de tumores de mama como malignos ou benignos a partir de características extraídas de imagens de biópsia por aspiração com agulha fina (FNA).

- Fonte: [Kaggle — UCI ML Breast Cancer Wisconsin](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
- 569 amostras | 30 features | Variável alvo: `diagnosis` (M = maligno, B = benigno)

## Estrutura do projeto

```
tech-challenge-1/
├── data/
│   └── processed/          # Dados pré-processados (gerados localmente)
├── notebooks/
│   ├── 01_eda.ipynb                  # Análise exploratória de dados
│   ├── 02_preprocessing.ipynb        # Pré-processamento e pipeline
│   └── 03_modeling.ipynb             # Modelagem, avaliação e SHAP
├── src/
│   ├── preprocessing.py    # Funções reutilizáveis de pré-processamento
│   └── evaluation.py       # Funções de avaliação e métricas
├── models/                 # Modelos treinados (gerados localmente)
├── reports/
│   └── figures/            # Gráficos e visualizações geradas
├── requirements.txt
└── README.md
```

## Modelos utilizados

| Modelo | Descrição |
|---|---|
| **Regressão Logística** | Baseline linear, rápido e interpretável |
| **Random Forest** | Ensemble de árvores, captura relações não-lineares |

## Resultados

| Modelo | Accuracy | Recall | Precision | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Regressão Logística | — | — | — | — | — |
| Random Forest | — | — | — | — | — |

> Os valores exatos são gerados ao executar o notebook `03_modeling.ipynb`.

## Métricas de avaliação

Dado o contexto médico, o **Recall** é a métrica mais crítica — um falso negativo (não detectar um tumor maligno) tem consequências graves para a paciente. São reportadas também accuracy, precision, F1-score e AUC-ROC.

## Explicabilidade

- **Feature Importance** (Random Forest): importância global de cada variável
- **SHAP Beeswarm**: impacto de cada feature em cada amostra individualmente
- **SHAP Waterfall**: explicação detalhada de um caso específico

## Como executar

### Pré-requisitos

- Python 3.10+ (recomendado via [Miniconda](https://docs.conda.io/en/latest/miniconda.html))

### Instalação

```bash
pip install -r requirements.txt
```

### Download do dataset

```bash
pip install kaggle
kaggle datasets download -d uciml/breast-cancer-wisconsin-data -p data/raw --unzip
```

### Execução

Execute os notebooks na ordem dentro de `notebooks/`:

```
01_eda.ipynb          → Análise exploratória
02_preprocessing.ipynb → Limpeza, encoding e pipeline
03_modeling.ipynb     → Treinamento, avaliação e SHAP
```

## Discussão crítica

Os modelos têm caráter de **apoio à decisão**, não de diagnóstico autônomo. O médico responsável deve sempre ter a palavra final, utilizando as predições como ferramenta auxiliar de triagem.

Limitações:
- O dataset Wisconsin é coletado em condições controladas; performance em dados reais pode variar
- Qualquer falso negativo representa risco clínico grave
- O modelo deve ser apresentado como "score de risco", não como diagnóstico definitivo
