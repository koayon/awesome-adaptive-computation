# Awesome Adaptive Computation

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Awesome Adaptive Computation is a curated list of Adaptive Computation papers, models, explainers and libraries for Machine Learning.

## Contents

- [Awesome Adaptive Computation](#awesome-adaptive-computation)
- [Contents](#contents)
- [About](#about)
- [Pre-Cursors](#pre-cursors-to-adaptive-computation)
  <!-- - [Algorithm](#algorithm) -->
  <!-- - [System](#system)
  - [Application](#application) -->
  <!-- - [Open-Source System](#open-source-system) -->
- [AI Safety](#ai-safety)

### About

`Adaptive Computation` is the ability of a machine learning system to adjust its `function` and `compute budget` for each example. We can think of this as giving models [System 2](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow) thinking.

[The Bitter Lesson]() LINK states that the scalable methods that should focus Machine Learning Research on are `Learning` and `Search`. Large pre-trained models focus traditionally on `learning` at _train time_; finetuning methods like RLHF are also about `learning`. `Search` on the other hand can be thought of as general approaches to get good performance by spending more compute at _inference_ time.

---

In this repo, links are organised by topic and have explanations so you can decide what you would like to read. Especially recommended links are starred 🌟

Star this repository to see the latest developments in this research field.

We accept contributions! We strongly encourage researchers to make a pull request with papers, approaches and explanations that they feel others in the community would benefit from 🤗

<!-- Ordered by topic, then date published -->

## Pre-cursors to Adaptive Computation

**Adaptive Mixtures of Local Experts, Jacobs et al (1991)**. [pdf](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)

> Collaborative, learned Mixture of Experts approaches to handle subsets of the training set are proposed.
> It's remarkable how close current approaches are to the original gating network.
> They also show intuitive expert specialisation on the task of vowel discrimination.

<!-- Chain of Thought

Beam Search -->

<!--

## Survey Papers

## Tools & Agents
One way of varying compute is on some tokens calling out to an external API to complete the token.

## Games

-->

<!-- ## Approaches We're Excited To See Explored More -->

## AI Safety

With adaptive computation, we want models to use more compute on harder problems.

For problems where we're concerned about systems failing by not being able to do sufficient computation then Adaptive Computation is very positive for Alignment. We should expect fewer mistakes from a model utilising Adaptive Computation, even on more difficult problems.

However, for problems where we're concerned about systems being deceptive or mesa-optimising increasing the ammount of inference-time compute increases their ability to do so. Here the failure is not a "mistake" but entirely intentional from the system's perspective.

<br>
<br>

---

<br>

Thanks for reading, if you have any suggestions or corrections please submit a pull request!
And please hit the star button to show your appreciation.

<!-- Soon:

MoE

**Designing Effective Sparse Expert Models, Google: Zoph et al (arXiv 2022)**. [pdf]

---
End to End

Universal Transformer, AUTHORS (2018) -
> Extends the idea of ACT to Transformers by using the number of Transformer Layers as the unit of variable compute. Followed up by PonderNet which refines the idea with a system that works better.

SkipNet: Dynamic Routing in CNNs, Wang et al (2017) [pdf](https://arxiv.org/pdf/1711.09485)

Spatially Adaptive Computation Time for Residual Networks ???

🌟 Adaptive Computation Time (ACT) for RNNs, Graves (2016)

Conditional Computation: Programmable Modulation of Deep Networks, Bengio et al. (2013)

The Early Exit Dilemma in Neural Network Training

-----
Review

A Review of Sparse Expert Models, Fedus et al (2022) [pdf], [video at Stanford], [podcast]

---
Tools

Toolformer

GPT-4 Plugins, OpenAI (2023) [blog], [demo]

---
Games

🌟 Cicero, Meta: AUTHORS (2019) [pdf]

AlphaGo [pdf], [film]

---

🌟 FrugalGPT

Tree of Thought

Asking follow ups? (Ofir Press?)

Debate

---
## Benchmarks

Parity

Complex logic questions

ContextQA location dataset

ARB (DuckAI benchmark)

Citation

-->
