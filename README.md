# Article Elo NLP

A weakly supervised NLP pipeline for estimating the political positioning of news articles using **web scraping, pairwise ranking, and BERT regression**.

The project converts coarse publication-level political-bias ratings into continuous article-level scores by generating pairwise comparisons and fitting a **Bradley–Terry ranking model**. Those scores are then used as regression targets for a partially fine-tuned BERT model that predicts ideological position directly from article text.

## Pipeline

```text
AllSides + Wayback Machine
            ↓
     Article Scraping
            ↓
 Source Rating Matching
            ↓ 
 Bradley–Terry Ranking
            ↓
 Continuous Weak Labels
            ↓
    BERT Regression
            ↓
Article-Level Bias Estimate
```

## Motivation

Political-bias datasets commonly assign publications broad labels such as **Left**, **Lean Left**, **Center**, **Lean Right**, or **Right**. These categories are useful at the publication level but provide limited information about variation between individual articles.

This project explores whether publication-level ratings can be transformed into more granular article-level targets through pairwise ranking, then learned directly from article text with a transformer model.

The result is an experimental pipeline connecting:

* web scraping and dataset construction
* entity/source matching
* pairwise comparison generation
* probabilistic ranking
* weak supervision
* transformer-based NLP regression

## Repository Structure

```text
article-elo-nlp/
│
├── 01_allsides_scraping.ipynb
├── 02_bradley_terry_ranking.ipynb
├── 03_bert_regression.ipynb
├── README.md
└── requirements.txt
```

### `01_allsides_scraping.ipynb`

Collects political news articles and source metadata from archived AllSides pages.

The scraper uses:

* **Playwright** for browser automation and dynamic page loading
* **BeautifulSoup** for HTML parsing
* **Wayback Machine** snapshots for archived AllSides and publisher pages
* randomized request delays during repeated scraping

The resulting article corpus is used for downstream ranking and NLP analysis.

### `02_bradley_terry_ranking.ipynb`

Constructs continuous article-level ideological scores from publication-level AllSides ratings.

The notebook:

1. normalizes publication names across datasets
2. applies manual aliases and fuzzy source matching
3. maps articles to AllSides ratings
4. determines pairwise ideological ordering
5. fits a Bradley–Terry model using `choix.ilsr_pairwise`
6. produces continuous latent scores for ranked articles

The resulting scores act as **weak-supervision labels** for the NLP model.

### `03_bert_regression.ipynb`

Trains a transformer model to predict the continuous ranking score directly from article text.

The architecture is:

```text
Article Text
     ↓
BERT Tokenizer
     ↓
bert-base-uncased
     ↓
[CLS] Representation
     ↓
Dropout
     ↓
Linear Regression Head
     ↓
Predicted Bias Score
```

The training pipeline includes:

* train / validation / test splitting
* standardized regression targets
* dynamic token padding
* partial BERT fine-tuning
* AdamW optimization
* Huber loss
* gradient clipping
* early stopping

## Pairwise Ranking

The ranking stage treats ideological positioning as a **relative comparison problem**.

Instead of immediately assigning every article an independent numerical label, two articles are compared at a time.

For articles from sources in different AllSides categories:

```text
Left < Lean Left < Center < Lean Right < Right
```

The article associated with the relatively more right-leaning source is treated as the winner of that comparison.

When two articles fall within the same categorical group, the more granular AllSides numerical source rating is used as a tie-breaker.

These winner/loser observations form a comparison graph used to estimate latent scores through a Bradley–Terry model.

Conceptually:

```text
Article A ──┐
            ├── Pairwise comparison ──→ Winner / Loser
Article B ──┘
                     ↓
      Bradley–Terry probability model
                     ↓
           Latent article scores
                     ↓
            BERT regression targets
```

### Bradley–Terry Model

The Bradley–Terry model assumes that each article $i$ has a latent score $\beta_i$. For a pair of articles $i$ and $j$, the probability that article $i$ is ranked above article $j$ is:

$$
P(i \succ j) =
\frac{e^{\beta_i}}
{e^{\beta_i} + e^{\beta_j}}
$$

Equivalently:

$$
P(i \succ j) =
\frac{1}
{1 + e^{\beta_j - \beta_i}}
$$

where:

* `β_i` is the latent ideological score of article `i`
* `β_j` is the latent ideological score of article `j`
* `P(i ≻ j)` is the probability that article `i` is ranked as relatively more right-leaning than article `j`

Across a large set of observed pairwise outcomes, the model estimates the latent `β` values that best explain the comparison results.

In this project, the pairwise outcomes are generated from AllSides source ratings. The resulting latent scores are then used as continuous weak-supervision targets for the BERT regression model.

## Machine Learning Model

The resulting ranking scores are treated as continuous regression targets.

The NLP model uses `bert-base-uncased` with a custom regression head. Most BERT parameters are frozen, while the final encoder layers are fine-tuned on the article dataset.

The model is evaluated using:

* **R²**

For the article dataset, the experiment recorded:

| Metric |     Result |
| ------ | ---------: |
| R²     | **0.7402** |

The held-out R² indicates that the BERT regression model captures a substantial portion of the variation in the generated article-ranking targets.

## Technologies

**Languages & Data**

`Python` · `pandas` · `NumPy`

**Web Scraping**

`Playwright` · `BeautifulSoup` · `Wayback Machine`

**Ranking & Statistics**

`choix` · `Bradley–Terry models` · `pairwise ranking`

**Machine Learning**

`PyTorch` · `Hugging Face Transformers` · `scikit-learn`

**NLP**

`BERT` · `tokenization` · `transformer embeddings` · `regression`

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/article-elo-nlp.git
cd article-elo-nlp
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

For the scraping notebook, install Chromium for Playwright:

```bash
playwright install chromium
```

Then open the notebooks with Jupyter:

```bash
jupyter notebook
```

Run the notebooks in order:

```text
01_allsides_scraping.ipynb
        ↓
02_bradley_terry_ranking.ipynb
        ↓
03_bert_regression.ipynb
```

## Limitations

The generated article scores should **not** be interpreted as objective measurements of political bias.

The most important limitation is that the pairwise outcomes originate from **publication-level AllSides ratings**, rather than independent human judgments of every individual article. The resulting article scores therefore represent **weak supervision** and can inherit assumptions or biases present in the source ratings.

Additional limitations include:

* publication-level ratings may not capture variation between articles from the same source
* archived webpages may be unavailable or inconsistently formatted
* publisher-specific HTML structures can introduce noise into scraped article text
* pair sampling affects the structure of the ranking graph
* political ideology is multidimensional and cannot be completely represented by a single continuous axis

The project is therefore best understood as an experiment in **weak supervision, pairwise ranking, and NLP modeling**, rather than an authoritative political-bias classifier.

## Project Focus

This project explores a broader question:

> **Can noisy, categorical labels be transformed into useful continuous supervision for NLP models?**

Political-news bias provides the test case, but the same architecture (**pairwise comparisons → latent ranking → learned text model**) can generalize to other subjective attributes where precise ground-truth labels are difficult or expensive to collect.
