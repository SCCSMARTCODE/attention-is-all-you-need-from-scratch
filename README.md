# Transformer from Scratch

A deep dive into implementing the Transformer architecture from scratch. This project provides a step-by-step implementation of the core components of the Transformer model, inspired by the original [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper.

---

## 📚 Overview

Transformers have revolutionized the fields of Natural Language Processing (NLP) and Computer Vision (CV) with their ability to model complex relationships in sequential data. This repository focuses on implementing the Transformer architecture from first principles to build a strong understanding of its internal workings.

---

## ✨ Features

- Complete implementation of the Transformer model, including:
  - Scaled dot-product attention
  - Multi-head attention mechanism
  - Positional encoding
  - Feedforward layers
- Modular and easy-to-read codebase.
- Compatible with PyTorch for seamless integration into other projects.

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/transformer-from-scratch.git
   cd transformer-from-scratch
   ```

2. Create a virtual environment:
   ```bash
   pip install pipenv
   pipenv install
   pipenv shell
   ```
---


### Model Components

Explore individual components:
- [Self-Attention Module](src/models/self_attention.py)
- [Positional Encoding](src/models/positional_encoding.py)
- [Multi-Head Attention](src/models/multihead_attention.py)

---

## 🧩 Code Structure

```
transformer-from-scratch/
│   .gitignore
│   LICENSE
│   Pipfile
│   Pipfile.lock
│   README.md
│
└───src
    ├───models
    │       decoder.py
    │       embedding.py
    │       encoder.py
    │       feedforward.py
    │       layer_normalization.py
    │       multihead_attention.py
    │       positional_encoding.py
    │       residual_connection.py
    │       transformer.py
    │       __init__.py
    │
    └───utils
            build.py
            __init__.py

```

---

## 🔧 Key Components

1. **Self-Attention**  
   Implementation of scaled dot-product attention, the core building block of the Transformer.

2. **Multi-Head Attention**  
   Combines multiple attention heads to capture information at different scales.

3. **Positional Encoding**  
   Encodes the relative positions of words in a sequence to inject order information.

4. **Feedforward Layers**  
   Fully connected layers applied to each token independently for further processing.

5. **Layer Normalization and Residual Connections**  
   Stabilizes training and helps with gradient flow.

---

## 🤝 Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, feel free to create an issue or submit a pull request.

---

## 🧑‍💻 Author

- **Your Name**  
  [EMMANUEL AYOBAMI ADEWUMI](https://github.com/SCCSMARTCODE) | [LinkedIn](https://www.linkedin.com/in/emmanuelayobami/)

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
