# 🏷️ AraT5 Hierarchical Text Classifier

[![Model](https://img.shields.io/badge/Model-UBC--NLP%2FAraT5--base-blue.svg)](https://huggingface.co/UBC-NLP/AraT5-base)
[![Libraries](https://img.shields.io/badge/Libraries-Transformers%20%7C%20PEFT%20%7C%20Datasets-orange.svg)](https://github.com/huggingface/transformers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Author](https://img.shields.io/badge/Developer-Eng.%20Bassem%20Al--Taheri-blueviolet.svg)](#-author)

A professional, state-of-the-art hierarchical text classification system for Arabic articles. The model is fine-tuned on the **Almersal Arabic Articles Dataset** using the **UBC-NLP/AraT5-base** seq2seq model, mapping titles and contents to a hierarchical category structure (`Main Category` → `Subcategory`).

---

## 👨‍💻 Author
**Eng. Bassem Al-Taheri**
- **GitHub:** [@BMA2U](https://github.com/BMA2U)
- **Scraper Pipeline:** [Almrsal-Web-Scraper-Dataset-Pipeline](https://github.com/BMA2U/Almrsal-Web-Scraper-Dataset-Pipeline)

---

## 📊 Data Source & Pipeline
The training dataset is scraped and compiled from the popular Arabic website *Almrsal* (المرسال). The complete data ingestion, processing, and pipeline creation are handled in the sister repository:
🔗 **[Almrsal-Web-Scraper-Dataset-Pipeline](https://github.com/BMA2U/Almrsal-Web-Scraper-Dataset-Pipeline)**

---

## 🎯 Task & Target Architecture
This project utilizes a **Seq2Seq generative classification** approach. Rather than training multiple classifiers or flat multi-label heads, the model learns to generate a hierarchical target sequence formatted with special tokens:

* **Input Prompt Format:**
  ```text
  المهمة: تصنيف هرمي

  العنوان:
  [Article Title]

  النص:
  [Article Content]
  ```

* **Target Label Format:**
  ```text
  <main_class>[Main_Category] <sub_class>[Subcategory]
  ```

Special tokens `<main_class>` and `<sub_class>` are added to the tokenizer vocabulary and embedding layer to preserve hierarchical boundary constraints.

---

## ⚡ Training Strategies
The notebook supports three training strategies depending on compute resources (specifically optimized for Google Colab Free Tesla T4 GPU):

| Strategy | Description | Trainable Params | VRAM Usage | Training Time (T4) |
|---|---|---|---|---|
| **LoRA (PEFT)** 🥇 | Injects trainable low-rank adapters into attention layers. | **~2-5%** | **~6 GB** | **3-5 hours** (Recommended) |
| **Freeze Encoder** 🥈 | Freezes the encoder and fine-tunes only the decoder. | **~50%** | **~10 GB** | **5-7 hours** |
| **Full Fine-tune** 🥉 | Trains all 220M parameters of the model. | **100%** | **~14 GB** | **24-36 hours** |

---

## ⚙️ Configuration & Hyperparameters
You can customize the training pipeline by editing the config block in the notebook:

```python
MODEL_NAME = "UBC-NLP/AraT5-base"
TRAINING_STRATEGY = "lora" # Options: "lora", "freeze_encoder", "full"

# Hyperparameters
NUM_EPOCHS = 4
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 16
SEED = 42

# LoRA Specifics
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
```

---

## 📋 Notebook Structure & Workflow

The main training notebook [`model_training_pro.ipynb`](./model_training_pro.ipynb) contains the following sections:

1. **Setup & Config:** Installs required dependencies (`transformers`, `peft`, `datasets`, etc.), mounts Google Drive, and verifies GPU capabilities (T4 checking).
2. **Data Preparation:** Conversions from raw JSON format into clean T5-ready JSONL files (`input` and `target` keys).
3. **Tokenization:** Instantiation of tokenizer, resizing vocabulary, adding special tokens, and caching datasets for faster execution.
4. **Model Setup:** Initializing model weight configuration, resizing embedding dimensions, and applying selected Parameter-Efficient Fine-Tuning (PEFT) strategy.
5. **Training Pipeline:** Configuring `Seq2SeqTrainer` with optimizations such as gradient checkpointing, mixed precision (`fp16`), and gradient accumulation.
6. **Hierarchical Evaluation:** Evaluating the model using structured metrics (Classification Report and Confusion Matrices) on generated label boundaries.
7. **Inference Engine:** A predictor function to classify unseen articles dynamically.
8. **Export & Publishing:** Saving checkpoints to Google Drive and optional publishing to Hugging Face Hub.

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
Ensure you have access to a GPU environment (Google Colab with T4 GPU or higher is recommended).

### 2. Setup Dependencies
Install the required packages in your environment:
```bash
pip install -U transformers datasets sentencepiece accelerate evaluate sacrebleu scikit-learn peft
```

### 3. File Layout
Place your raw dataset files under your Google Drive directory at `MyDrive/almersal_dataset/`:
* `train.json`
* `validation.json`
* `test.json`

### 4. Running the Code
Open [`model_training_pro.ipynb`](./model_training_pro.ipynb) in Google Colab, select your desired `TRAINING_STRATEGY`, and run all cells.

---

## 📈 Optimization Highlights
* **VRAM Savings:** Utilizes `gradient_checkpointing` which reduces memory consumption by ~40% during backpropagation.
* **Effective Batch Size:** Combines a physical batch size of `8` (with LoRA) and `gradient_accumulation_steps=2` to simulate a stable batch size of `16`.
* **Reproducibility:** Seeded random generator initializations (`seed=42`) across NumPy, PyTorch, and random environments.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
