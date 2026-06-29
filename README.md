# Tech Challenge Fase 1 — IA para Saúde Feminina

Sistema inteligente de suporte ao diagnóstico médico com foco na saúde e segurança da mulher, utilizando Machine Learning para identificação precoce de câncer de mama.

## Objetivo

Construir modelos preditivos de classificação sobre dados médicos relacionados à saúde feminina, aplicando técnicas de Machine Learning, análise exploratória e explicabilidade dos resultados.

## Datasets

### Tarefa principal — dados estruturados
**Breast Cancer Wisconsin Diagnostic Dataset**
Classificação de tumores de mama como malignos ou benignos a partir de características extraídas de biópsias (FNA).

- Fonte: [Kaggle — UCI ML Breast Cancer Wisconsin](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
- 569 amostras | 30 features | Variável alvo: `diagnosis` (M = maligno, B = benigno)

### Extra — visão computacional
**Breast Ultrasound Images Dataset (BUSI)**
Classificação de ultrassonografias de mama em três categorias: benigno, maligno e normal.

- Fonte: [Kaggle — Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- 780 imagens | 3 classes: benigno (437), maligno (210), normal (133)

## Estrutura do projeto

```
tech-challenge-1/
├── data/
│   ├── raw/
│   │   ├── data.csv                       # Dataset Wisconsin (estruturado)
│   │   └── images/Dataset_BUSI_with_GT/   # Imagens BUSI (baixar via Kaggle)
│   └── processed/                         # Dados pré-processados (gerados localmente)
├── notebooks/
│   ├── 01_eda.ipynb                  # Análise exploratória de dados
│   ├── 02_preprocessing.ipynb        # Pré-processamento e pipeline
│   ├── 03_modeling.ipynb             # Modelagem ML, avaliação e SHAP
│   └── 04_cnn.ipynb                  # [EXTRA] CNN com Transfer Learning (VGG16)
├── src/
│   ├── preprocessing.py    # Funções reutilizáveis de pré-processamento
│   └── evaluation.py       # Funções de avaliação e métricas
├── models/                 # Modelos treinados (gerados localmente)
├── reports/
│   └── figures/            # Gráficos e visualizações geradas (01–20)
├── requirements.txt
└── README.md
```

## Modelos utilizados

| Notebook | Modelo | Descrição |
|---|---|---|
| `03_modeling` | **Regressão Logística** | Baseline linear, rápido e interpretável |
| `03_modeling` | **Random Forest** | Ensemble de árvores, captura relações não-lineares |
| `04_cnn` (extra) | **CNN VGG16** | Transfer Learning para classificação de ultrassonografias |

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

### Download dos datasets

```bash
pip install kaggle

# Dataset Wisconsin (dados estruturados)
kaggle datasets download -d uciml/breast-cancer-wisconsin-data -p data/raw --unzip

# Dataset BUSI (imagens para CNN — extra)
kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset -p data/raw/images --unzip
```

### Execução

Execute os notebooks na ordem dentro de `notebooks/`:

```
01_eda.ipynb           → Análise exploratória
02_preprocessing.ipynb → Limpeza, encoding e pipeline
03_modeling.ipynb      → Treinamento, avaliação e SHAP
04_cnn.ipynb           → [EXTRA] CNN com Transfer Learning (requer TensorFlow)
```

## Discussão crítica

Os modelos têm caráter de **apoio à decisão**, não de diagnóstico autônomo. O médico responsável deve sempre ter a palavra final, utilizando as predições como ferramenta auxiliar de triagem.

Limitações:
- O dataset Wisconsin é coletado em condições controladas; performance em dados reais pode variar
- Qualquer falso negativo representa risco clínico grave
- O modelo deve ser apresentado como "score de risco", não como diagnóstico definitivo
