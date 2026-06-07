# D-Tree: A New Association Rule Mining Technique

## Overview

D-Tree (Dominant Tree) is a novel Association Rule Mining (ARM) algorithm developed as part of undergraduate research at Green University of Bangladesh.

The algorithm introduces three key concepts:

* InitSet (grouping similar transactions before tree construction)
* Dominant Node selection
* Non-Repeated Values (NRV) based insertion strategy

These techniques aim to improve the relevance of generated association rules while maintaining efficient execution time and memory usage.

---

## Research Contribution

Traditional ARM algorithms such as Apriori, FP-Growth, and EFP-Growth focus primarily on reducing computational complexity.

D-Tree introduces a new tree construction strategy that:

* Groups identical transactions before insertion
* Uses Dominant Nodes to guide tree construction
* Employs NRV to prioritize node insertion
* Produces more relevant association rules

---

## Algorithm Workflow

Database
→ Preprocessing
→ InitSet Generation
→ Dominant Node Identification
→ D-Tree Construction
→ Frequent Pattern Mining
→ Association Rule Generation

---

## Datasets

Included benchmark datasets:

* Mushroom
* Zoo
* Grocery
* MBO

---

## Repository Structure

data/

src/

paper/

results/


---

## Installation

pip install -r requirements.txt

---

## Run

python src/d_tree.py

---

## Experimental Results

According to the evaluation reported in the paper, D-Tree achieved:

* Higher Accuracy than FP-Growth and EFP-Growth
* Higher Sensitivity
* Higher AUC
* Lower execution time on benchmark datasets
* Lower memory consumption in several scenarios

---

## Paper

The complete paper is available in:

https://www.researchgate.net/publication/396342402_A_New_Association_Rule_Mining_Technique

---

## Authors

Abdullah Al Maruf

Montasir Rahman

Umme Karima Oyshi

Saiful Azad

M. Solaiman Mia

---

## Citation

If you use this work in your research, please cite:

A New Association Rule Mining Technique (D-Tree)
