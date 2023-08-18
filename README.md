# Awesome Adaptive Computation

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Awesome Adaptive Computation is a curated list of Adaptive Computation papers, models, explainers and libraries for Machine Learning.

## Contents

- [Awesome Adaptive Computation](#awesome-adaptive-computation)
  - [Contents](#contents)
  - [About](#about)
- [Pre-Cursors](#pre-cursors-to-adaptive-computation)
- [End-to-End Adaptive Computation](#end-to-end-adaptive-computation)
- [Black Box Adaptive Computation](#black-box-adaptive-computation)
- [Mixture of Experts](#mixture-of-experts)
- [Open Source Librarues](#open-source-libraries)
- [AI Safety](#ai-safety)
<!--  -->

### About

`Adaptive Computation` is the ability of a machine learning system to adjust its `function` and `compute budget` for each example. We can think of this as giving models [System 2](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow) thinking.

[The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) LINK states that the scalable methods that should focus Machine Learning Research on are `Learning` and `Search`. Large pre-trained models focus traditionally on `learning` at _train time_; finetuning methods like RLHF are also about `learning`. `Search` on the other hand can be thought of as general approaches to get good performance by spending more compute at _inference time_.

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

## End-to-End Adaptive Computation

Adaptive Computation with Elastic Input Sequence (AdaTape-ViT), Google: AUTHORS (2023) [pdf](https://arxiv.org/pdf/2301.13195.pdf), [blog](https://ai.googleblog.com/2023/08/adatape-foundation-model-with-adaptive.html), [official code](https://github.com/google-research/scenic/blob/main/scenic/projects/adatape/adatape_vit/adatape_vit.py)

> Extending the ACT method by giving the model a "tape" which contains some inputs which may be useful for encoding as well as the input.
> At each layer, the model can append a variable number of tape tokens to the input for processing which allows it to regulate how much additional compute we add.
> The paper shows impressive performs on image classification tasks and the 'parity' task on long sequences.

🌟 PonderNet, DeepMind: Banino et al (2021) [pdf](https://arxiv.org/pdf/2107.05407.pdf), [pytorch code](https://github.com/koayon/ml-replications/tree/main/ponder)

> Allows the model to exit after each transformer layer if it's confident in the answer.
> It introduces a stable probabilistic policy for halting which provides low-variance unbiased gradient updates.

## Black-box Adaptive Computation

Here we mean techniques that you could use with an already trained model where you get either only the output tokens or you get the final layer logits. No retraining is required and therefore these are promising techniques for people with limited training compute budgets.

🌟 **Speculative Sampling, DeepMind: Chen et al (2023)** [pdf](https://arxiv.org/pdf/2302.01318.pdf), [pdf2](https://arxiv.org/pdf/2211.17192.pdf), [blog](https://jaykmody.com/blog/speculative-sampling/), [code](https://github.com/jaymody/speculative-sampling)

> A smaller model does the autoregressive generation for multiple tokens and then a larger model checks the smaller model against what it would have generated in one go. We accept only the tokens where the two models agree and then the larger model's next token.
> This gives exactly the same output as the larger model would have but with significantly reduced sampling time.

## Mixture of Experts

🌟 Expert Choice MoEs, Google: Zhou et al (2022) [pdf](https://arxiv.org/pdf/2202.09368.pdf), [blog](https://ai.googleblog.com/2022/11/mixture-of-experts-with-expert-choice.html), [pytorch code](https://github.com/koayon/ml-replications/blob/main/mixture_of_experts/expert_choice_layer.py)

> Introduces the first truly adaptive computation MoE model.
> In traditional MoE models the tokens select the top experts that they would most like to be processed by. In Expert Choice routing however, the experts choose the top tokens that they would like to process. Hence multiple experts can pick the same token and give it lots of compute, and similarly all experts can ignore a token so it is skipped for that layer.
> As well as improving training efficiency, this approach also has the benefits that it helps with load balancing.

Switch Transformers, Google: Fedus et al (2021) [pdf](https://arxiv.org/pdf/2101.03961.pdf), [pytorch code](https://nn.labml.ai/transformers/switch/index.html), [model](https://huggingface.co/docs/transformers/model_doc/switch_transformers)

> Simplifies the MoE routing algorithm with top-1 routing. Shows that we can exploit the scaling laws with parameters as well as simply compute and develops distrbuted systems approach to MoE.

🌟 Outrageously Large Neural Networks (aka The Sparse MoE Layer), Google: Shazeer et al (2017) [pdf](https://arxiv.org/pdf/1701.06538.pdf)

> Introduces Mixture of Expert models in their modern form using Sparsely Gated MoE layer and a trainable gating network.
> They use RNNs as this is pre Transformers Eating The World.

## Open Source Libraries

🌟 DeepSpeed-MoE, Microsoft: Rajbhandari et al (2022) [blog](https://www.microsoft.com/en-us/research/blog/deepspeed-advancing-moe-inference-and-training-to-power-next-generation-ai-scale/), [pdf](https://arxiv.org/pdf/2201.05596.pdf), [code](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/moe)

> Training solution and inference solution for distributed MoE models as part of the DeepSpeed library. Improves training efficiency and serving latency.
> They also present a new MoE architecture PR-MoE which is has more experts in higher layers and a method for distilling expert models into dense 'student models'.

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

Black box

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
