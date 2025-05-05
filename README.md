## 🧠 Transformer-Based Sentiment Classifier

This project is a simple, custom-built **Transformer-based sentiment analysis model** that classifies user feedback (e.g., reviews of products, services, apps) as **positive** or **negative**. The model is implemented from scratch using PyTorch and runs efficiently on both CPU and GPU.

---

### 🚀 Features

* Transformer architecture with multi-head self-attention
* Custom vocabulary and word embedding
* Dropout, fully connected layers, and output projection
* `BCEWithLogitsLoss` for binary classification
* GPU acceleration with CUDA support
* Real-time train/test accuracy reporting

---

### 🛠 Requirements

Install the following Python packages:

```bash
pip install torch scikit-learn
```

---

### 📁 Project Structure

```
.
├── transformers_deepLearning.py   # Full model and training pipeline
└── README.md                      # Project documentation
```

---

### ⚙️ How to Run

1. Open your terminal or code editor.
2. Run the script with:

```bash
python transformers_deepLearning.py
```

If a GPU is available, the model will use it automatically via PyTorch's CUDA interface.

---

### 📊 Model Architecture Summary

* **Embedding Dimension**: 32
* **Number of Heads**: 4
* **Encoder Layers**: 2
* **Maximum Sequence Length**: 15
* **Hidden Layer Size**: 64
* **Dropout**: 0.3
* **Optimizer**: Adam
* **Loss Function**: BCEWithLogitsLoss
* **Learning Rate Scheduler**: StepLR (decays every 5 epochs)

---

### ✅ Example Output

```
Epoch 1/50 | Loss: 0.6935
...
Epoch 50/50 | Loss: 0.2317

✅ Test Accuracy : 0.8750
✅ Train Accuracy: 0.9375
```

---

### 💡 Tips for Improvement

* You can improve performance by increasing the dataset size.
* Experiment with hyperparameters like `max_len`, `embedding_dim`, and `hidden_dim`.
* Replace the custom tokenizer and model with pretrained models like BERT for production-level results.

---

### 🧩 License

This project is intended for educational and research purposes. Contact the author for commercial use.

---


