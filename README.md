# 🧬 Three-Way Multiple Sequence Alignment (MSA) in Python

[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)
[![Numba Accelerated](https://img.shields.io/badge/numba-optimized-yellow.svg)](https://numba.pydata.org/)

---

## 📑 Overview

This repository provides an efficient, **Numba-accelerated implementation of multiple sequence alignment (MSA) for three DNA sequences**, using **dynamic programming** and a **divide-and-conquer** strategy.  
It is designed for rapid, memory-efficient alignment of three sequences with full traceback and customizable scoring.

---

## 🚀 Features

- **Three-way (3D) sequence alignment** with full DP and divide-and-conquer recursion.
- **Impulse-fast**: Core DP kernels accelerated with [Numba](https://numba.pydata.org/).
- **Customizable scoring function** (sum-of-pairs).
- **Memory monitoring** during alignment.
- **Simple input/output**: edit input strings or provide your own files.
- Generates and reads FASTA files automatically.

---

## 🧠 Background & Method

- **Multiple Sequence Alignment (MSA)** is a fundamental task in computational biology, aligning three or more biological sequences (DNA, RNA, protein) to identify regions of similarity.
- This code supports **three-way alignment** using a full 3D dynamic programming (Needleman-Wunsch) algorithm with efficient **divide-and-conquer** recursion to reduce memory usage for large sequences.
- The default scoring uses sum-of-pairs: +5 for match, -4 for mismatch, -8 for gaps, but you can provide your own.

---

## 🏗️ Installation

**Requirements:**
- Python 3.7+
- [numpy](https://numpy.org/)
- [numba](https://numba.pydata.org/)
- [biopython](https://biopython.org/)
- [psutil](https://github.com/giampaolo/psutil)

Install all dependencies with:

```bash
pip install numpy numba biopython psutil
```

## ▶️ How to Run

After installing the required packages, you can run the three-way MSA algorithm by executing the script from the command line.  
**Example:**

```bash
python MSA.py
```
