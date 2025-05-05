---
# You can also start simply with 'default'
theme: seriph
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://cdn.jsdelivr.net/gh/slidevjs/slidev-covers@main/static/nC_dpX5Q_bA.webp
# some information about your slides (markdown enabled)
title: KnowHiRA
info: Knowledge-aware Hadamard-integrated Rank Adaptation
# apply unocss classes to the current slide
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: fade-out
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true
# open graph
# seoMeta:
#  ogImage: https://cover.sli.dev
---

# KnowHiRA

## Knowledge-aware Hadamard-integrated Rank Adaptation for Efficient Commonsense Reasoning

<div class="pt-10 pb-5">
  <div class="flex justify-center gap-8 mt-2">
    <div>
      <strong>Yi Cui</strong>
    </div>
    <div>
      <strong>Yihe Pan</strong>
    </div>
    <div>
      <strong>XuanYi Yang</strong>
    </div>
  </div>
</div>

---
layout: center
---

# Table of Contents

<Toc text-sm minDepth="1" maxDepth="1" />

::right::

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---
layout: two-cols
layoutClass: gap-16
---

# Problem Statement

## **Background**

- Fine-tuning LLMs is computationally expensive
- Parameter-efficient fine-tuning (PEFT) addresses this challenge
- New PEFT Methods
  - **HiRA**: High-Rank Adaptation using Hadamard products
  - **KaSA**: Knowledge-aware Singular-value Adaptation

::right::

## **Target Task**

- Testing parameter-efficient fine-tuning quality and efficiency
- Requires both expressivity and knowledge alignment
- Challenges models to understand everyday situations and implicit knowledge

## **Our Approach**

**KnowHiRA** enhances **HiRA** with knowledge-aware mechanisms from **KaSA** for efficient commonsense reasoning.

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---
layout: two-cols
layoutClass: gap-16
---

# Dataset

## **Dataset Statistics**

<div class="text-xs">

| Dataset | Data Number | Type |
|---------|-------------|------|
| **Commonsense-170K** | 170,300/120 | Mixed |
| BoolQ | 3,270 | Yes/No |
| PIQA | 1,830 | Option |
| SIQA | 1,954 | Option |
| HellaSwag | 10,042 | Option |
| WinoGrande | 1,267 | Option |
| ARC-e | 2,376 | Option |
| ARC-c | 1,172 | Option |
| OBQA | 500 | Option |

</div>

::right::

<span style="color: #2B90B6">**Fine-tuning Dataset: Commonsense-170K**</span>

- 170,420 query-answer pairs for fine-tuning
- 120 random entries as validation set

<span style="color: #2B90B6">**Evaluation Tasks:**</span>

- **BoolQ**: Yes/No question answering
- **PIQA**: Physical commonsense reasoning
- **SIQA**: Social interaction reasoning
- **HellaSwag**: Commonsense natural language inference
- **WinoGrande**: Fill-in-the-blank reasoning
- **ARC-e/c**: Multiple-choice science questions
- **OBQA**: Multi-step reasoning

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---
layout: two-cols
layoutClass: gap-16
---

# Background

## **HiRA**
  $$\Delta W = W_0 \odot (AB), \quad W = W_0 + \Delta W$$

<img src="/hira.png"/>

- **Rank Enhancement**: Leverages Hadamard product to increase rank upper bound
  $$\mathrm{Rank}(\Delta W) \leq \mathrm{Rank}(W_0) \times \mathrm{Rank}(AB)$$

::right::

- **Advantages**:
  - Maintains same parameter count as LoRA: $r(d+k)$ instead of $dk$
  - High expressivity: rank upper bound up to $r_0 \times r$
  - Gradients leverage prior information in pre-trained weights
- **Limitations**:
  - **Numerical Stability**: Hadamard product may introduce larger gradient scale fluctuations
  - **Zero-value Problem**: If $W_0$ contains zero entries, $\Delta W_{ij}=0$ at those positions

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---
layout: two-cols
layoutClass: gap-16
---

## **KaSA**

- **Knowledge-aware Singular-value Adaptation**: Dynamic knowledge activation approach
- **Two-stage Process**:
  1. **Knowledge-based SVD Truncation**: 
  $$W_{\text{world}} = U[:m-r]\Sigma[:m-r]V[:m-r]^\top$$
  2. **Knowledge-aware Singular-Value Adaptation**: 
  $$\Delta W = \Delta U\;\Delta\Sigma\;\Delta V^\top$$
- **Reasoning Weight**:
  $$W' = W_{\text{world}} + \eta\,\Delta U\,\Delta\Sigma\,\Delta V^\top$$

::right::

- **Advantages**:
  - Filters out noise/long-tail knowledge through SVD truncation
  - Dynamically activates task-relevant knowledge components
  - Maintains orthogonality constraints: $\Delta U^\top\Delta U=\Delta V^\top\Delta V=I_r$
- **Limitations**:
  - Sensitivity to SVD truncation hyperparameter $r$
  - Numerical stability issues with $\Delta\Sigma$ and orthogonal regularization
  - Limited to low-rank adaptation space

---
layout: center
---

<img src="/kasa.png"/>

---
layout: center
---

# Our Approach: KnowHiRA

<img src="/knowhira.png"/>

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---
layout: two-cols
layoutClass: gap-16
---

## **Core Methods**

<span style="color: #2B90B6">**Knowledge-guided Hadamard Updates**</span>

We reformulate **HiRA**'s update by incorporating knowledge from SVD of pre-trained weights:

$$\Delta W = W_0 \odot (A \cdot G_{\Sigma} \cdot B)$$

Where $G_{\Sigma}$ is a knowledge-guided gating matrix derived from singular values.

<span style="color: #2B90B6">**Adaptive Knowledge Gating**</span>

A learnable gating mechanism to dynamically control knowledge influence:

$$G_{\Sigma} = \text{Diag}(\sigma_{\text{norm}} \odot \text{sigmoid}(g))$$

::right::

<span style="color: #2B90B6">**Orthogonal Parameter Regularization**</span>

Maximizes effective rank while maintaining parameter efficiency:

$$L_{\text{ortho}} = \|A^T A - I\|_F^2 + \|B B^T - I\|_F^2$$

Our final training objective:

$$L = L_{\text{task}} + \lambda_{\text{ortho}} L_{\text{ortho}}$$

<span style="color: #2B90B6">**Spectrum-aware Initialization**</span> 

Leverages singular value patterns for faster convergence:

$$A_{init} = \text{Kaiming}(d, r) \cdot f(\Sigma), \\ 
\quad B_{init} \approx 0, \quad g_{init} < 0$$

Where $f(\Sigma)$ ensures initial updates respect the model's knowledge hierarchy.

---
layout: two-cols
layoutClass: gap-16
---

# Experimental Setup

- **Fine-tuning**: Commonsense-170K dataset (~170K samples)
- **Evaluation**: Benchmark suite & protocol profollowing HiRA
  - Social IQA, PIQA, BoolQ, HellaSwag
  - WinoGrande, ARC-Easy/Challenge, OpenbookQA
- **Models**: Pre-trained LLMs in 7B-13B parameter range
::right::

<span style="color: #2B90B6">**Baselines**</span>

| Method | Key Feature |
|--------|-------------|
| Full Fine-tuning | 100% parameters |
| LoRA | Low-rank adaptation |
| HiRA | Hadamard multiplicative updates |
| KaSA | SVD-based knowledge guidance |

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---
layout: center
---

# Work Plan

| Phase | Timeline | Key Activities |
|-------|----------|----------------|
| **Implementation** | Week 12-13 | Implement KnowHiRA method & set up experimental framework |
| **Experiments** | Week 13-14 | Fine-tune on Commonsense-170K & evaluate on benchmark datasets |
| **Analysis** | Week 13-14 | Compare with baselines & perform ablation studies |
| **Paper Writing** | Week 14-15 | Complete final paper |

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---
layout: center
class: text-center
---

# Thank you!

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

