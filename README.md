# English-to-Tamil-Translation using Transformer

This project implements a Transformer model for Neural Machine Translation (NMT) from English to Tamil. The model is built using PyTorch and is based on the architecture described in the paper "Attention Is All You Need".

## Overview

The goal of this project is to build and train a sequence-to-sequence Transformer model to translate English sentences into Tamil. The process involves:

1.  **Data Preparation:** Loading and preprocessing the parallel English-Tamil corpus.
2.  **Model Training:** Training the Transformer model on the prepared dataset.
3.  **Inference:** Using the trained model to translate new English sentences into Tamil.

## Dataset

The dataset used for this project is a parallel corpus of English and Tamil sentences. The data is split into training, validation, and test sets.

-   **Source Language:** English (`.en` files)
-   **Target Language:** Tamil (`.ta` files)

### Preprocessing

The text data is preprocessed using Byte Pair Encoding (BPE) with the `youtokentome` library. A shared BPE model (`bpe.model`) is trained on the training data to create a vocabulary of subword tokens. This helps the model handle out-of-vocabulary words and create a more efficient vocabulary.

## Model Architecture

The model is a standard Transformer architecture with an encoder-decoder structure.

-   **Encoder:** The encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn).
-   **Decoder:** The decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time, using the previously generated symbols as additional input when generating the next.

### Hyperparameters

The model was configured with the following hyperparameters:

| Hyperparameter      | Value  |
| ------------------- | ------ |
| `d_model`           | 256    |
| `n_layers`          | 3      |
| `n_heads`           | 4      |
| `d_queries`         | 64     |
| `d_values`          | 64     |
| `d_inner`           | 1024   |
| `dropout`           | 0.2    |

## Training

The model was trained using the following configuration:

-   **Optimizer:** Adam with `betas=(0.9, 0.98)` and `epsilon=1e-9`.
-   **Learning Rate:** A custom learning rate schedule with `warmup_steps=4000`.
-   **Loss Function:** Cross-Entropy Loss with Label Smoothing (`eps=0.1`).
-   **Batch Size:** Batches were created to contain approximately `2000` tokens.
-   **Gradient Accumulation:** Gradients were accumulated over `25000 // 2000` steps.

## Evaluation

The model's performance is evaluated using the BLEU (Bilingual Evaluation Understudy) score, which measures the similarity between the machine-translated text and a set of high-quality reference translations.

The `eval.py` script is used to calculate the BLEU score on the test set. It uses the `sacrebleu` library to compute the scores with different tokenization methods.

## Results

The following BLEU scores were achieved on the test set:

| Tokenization                 | Cased  | Caseless |
| ---------------------------- | ------ | -------- |
| **13a**                      | 9.53   | 9.54     |
| **International**            | 9.39   | 9.40     |

-   **`13a` tokenization, cased:** `BLEU = 9.53 38.4/14.0/6.8/3.8 (BP = 0.878 ratio = 0.885 hyp_len = 135010 ref_len = 152548)`
-   **`13a` tokenization, caseless:** `BLEU = 9.54 38.5/14.0/6.8/3.8 (BP = 0.878 ratio = 0.885 hyp_len = 135010 ref_len = 152548)`
-   **International tokenization, cased:** `BLEU = 9.39 38.6/14.0/6.7/3.7 (BP = 0.872 ratio = 0.880 hyp_len = 135458 ref_len = 153934)`
-   **International tokenization, caseless:** `BLEU = 9.40 38.6/14.0/6.7/3.7 (BP = 0.872 ratio = 0.880 hyp_len = 135458 ref_len = 153934)`

## How to Use

### Command-line Translation

To translate an English sentence to Tamil using the command-line interface, you can run the `translator.py` script:

```bash
python app/translator.py
```

The script will prompt you to enter an English sentence and will then output the best translation found using beam search.

### Local Web Application

To run the web application locally:

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the application:**
    ```bash
    python -m app.main
    ```
    This will start the FastAPI server, typically accessible at `http://127.0.0.1:8000`. Open this URL in your web browser to interact with the translation interface.

