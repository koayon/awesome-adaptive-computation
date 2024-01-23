# Awesome Adaptive Computation

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Awesome Adaptive Computation is a curated list of Adaptive Computation papers,
models, explainers and libraries for Machine Learning.

## Contents

- [Contents](#contents)
- [About](#about)
- [Mixture of Experts (Sparse MoE)](#mixture-of-experts-sparse-moe)
- [Other Modular Architectures](#other-modular-architectures)
- [Early Exit: End-to-End Adaptive Computation](#early-exit-end-to-end-adaptive-computation)
- [Adaptive Computation for Black-box models](#adaptive-computation-for-black-box-models)
- [Continual Learning](#continual-learning)
- [Tools \& Agents](#tools--agents)
- [Games](#games)
- [Pre-cursors to Adaptive Computation](#pre-cursors-to-adaptive-computation)
- [Open Source Libraries](#open-source-libraries)
- [AI Safety](#ai-safety)
- [Other](#other)

## About

`Adaptive Computation` is the ability of a machine learning system to adjust its
`function` and `compute budget` for each example.

<!-- We can think of this as giving models a [System 2](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow) mode. -->

Adaptive Computation techniques include **Mixture of Experts** (decoupling model
capacity and model compute) and **Early Exiting** (saving compute on easy
inputs) as well as sampling techniques.

<!-- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) states that the two general methods to utilise large amounts of compute are `Learning` and `Search`. Whilst large pre-trained models focus on `learning` at _train time_, Adaptive Computation is about spending more compute at inference time with mechanisms similar to `Search`. -->

---

In this repo, links are organised by topic and have explanations so you can
decide what you would like to read. Especially recommended links are starred 🌟

Star this repository to see the latest developments in this research field.

We accept contributions! We strongly encourage researchers & practitioners to
make pull requests with papers, approaches and explanations that they feel
others in the community would benefit from 🤗

<!-- Ordered by topic, then date published -->

## Mixture of Experts (Sparse MoE)

The MoE paradigm uses a routing layer to choose a limited number of parameters
to apply to a given input rather than using all the available parameters.

This conditional computation allows us model capacity to increase without also
scaling the compute required for each forward pass. This is useful because
bigger models are more sample efficient and more compute efficient to train.

MoE models are also useful for compartmentalising knowledge and avoiding
negative interference from irrelevant computation.
[Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) is an
open-weights MoE model which is comparable to much larger models.

**Mixtral of Experts, Mixtral (2024)**
[pdf](https://arxiv.org/pdf/2401.04088.pdf)
[official code](https://github.com/mistralai/mistral-src)

> The paper describing Mixtral's State of Art LLM based on the MoE paradigm.

**MoE Mamba, IDEAS NCBR: Pioro et al (2024)**
[pdf](https://arxiv.org/pdf/2401.04081.pdf)

> The authors combine the MoE paradigm with a recent SSM-based architecture
> Mamba. Mamba provides Transformer-like performance and scaling properties
> whilst using a sub-quadratic attention variant allowing for much larger
> sequence lengths. Here we see that this architecture can additionally be
> combined with MoEs to increase performance, similarly to MoEs for transformers
> or RNNs previously.

**Offloading for Fast MoE Inference, Moscow: Eliseev & Mazur (2023)**
[pdf](https://arxiv.org/pdf/2312.17238.pdf)
[official code](https://github.com/dvmazur/mixtral-offloading)

> Despite some open-weights MoEs being available, they are not the most popular
> models used for inference due to the large memory footprint required at
> inference time. To address this the authors propose architecture-aware
> quantisation, an LRU cache for experts (to exploit the fact that experts are
> more likely than chance to repeat for two adjacent tokens) and a speculative
> expert loading algorithm. Since the inputs (x) to each layer only differ
> iteratively by a small amount (due to the residual stream carrying information
> from layer to layer), they note that by applying the routing function for the
> subsequent layer at the current layer, you can get a good guess for which
> experts to load. The upshot of this is that MoE models, like Mixtral, can be
> run on consumer grade hardware with much increased generation speed. This is a
> huge win for inference efficiency in the memory constrained and single batch
> regime.

**QMoE, ISTA: Frantar & Alistarh (2023)**
[pdf](https://arxiv.org/pdf/2310.16795.pdf)
[code](https://github.com/IST-DASLab/qmoe)

> Generally MoEs require larger footprints but fewer FLOPs compared to a dense
> model of which achieves similar performance. In the quest to reduce the memory
> footprint, we might seek to perform quantisation. The authors present a
> compression method which takes advantage of the inherent sparsity to compress
> the model at a 20x compression rate whilst retaining most performance. This
> compresses each fp16 weight to the equivalent of less than one bit. For the
> first time it's possible to run a trillion parameter model on consumer
> hardware.

**SparseMixer - Sparse Backpropagation for MoE Training, Microsoft: Liu et al
(2023)** [pdf](https://arxiv.org/pdf/2310.00811.pdf),

> One of the most important parts of an MoE is the router which allows the
> experts to specialise well. Unfortunately, typical MoE training gives
> suboptimal routers as suggested by Hash routers performing almost as well as
> more principled routing mechanisms. This paper suggests the reason is due to
> MoE training ignoring parts of the gradient and suggests a midpoint-rule based
> gradient approximation which substantially improves training.

**MoV/MoLoRA, Cohere For AI: Zadouri et al (2023)**
[pdf](https://arxiv.org/pdf/2309.05444.pdf),
[official Jax code](https://github.com/for-ai/parameter-efficient-moe)

> Introduces parameter efficient MoE models where instead of routing between
> entire FFN layers, we route between adapters such as LoRAs or $(IA)^3$ with
> the same base model. This allows for much of the benefits of the (Soft) MoE
> paradigm but without the huge memory footprint (particularly compared to
> previous upscaling methods). The
> [HydraMoE](https://github.com/SkunkworksAI/hydra-moe) project also take a
> similar approach.

**Soft Merging of Experts (SMEAR), UNC: Zadouri et al (2023)**
[pdf](https://arxiv.org/pdf/2306.03745.pdf)

> Takes the opposite approach to Soft-MoE and averages the Expert weights rather
> than the tokens. This is interesting given that model merging approaches which
> show that linear combinations of models can perform well on tasks that either
> model was trained for. Note here the averaging operation can become
> prohibitively expensive if we different experts for each token (similar FLOPs
> to forward passes on all experts and ensembling). Hence the method relies on a
> Task-MoE approach of picking the same expert configuration on a per-example
> rather than a per-token basis.

**AutoMoE, UBC/Microsoft: Jawahar et al (2023)**
[pdf](https://arxiv.org/pdf/2210.07535.pdf),
[official PyTorch code](https://github.com/microsoft/AutoMoE)

> One of the promises of MoE is being able to apply different amounts of compute
> to each token. Generally, this has been achieved by different tokens being
> processed and dropped by different numbers of experts per layer. AutoMoE also
> uses differently sized experts to achieve more heterogeneity. They perform an
> Architectural search for optimal architectures given computational
> constraints.

🌟 **Expert Choice MoEs, Google: Zhou et al (2022)**
[pdf](https://arxiv.org/pdf/2202.09368.pdf),
[blog](https://ai.googleblog.com/2022/11/mixture-of-experts-with-expert-choice.html),
[PyTorch code](https://github.com/koayon/ml-replications/blob/main/mixture_of_experts/expert_choice_layer.py)

> Introduces a principled, truly compute-adaptive MoE model. In traditional MoE
> models the tokens select the top experts that they would most like to be
> processed by. In Expert Choice routing however, the experts choose the top
> tokens that they would like to process. Hence multiple experts can pick the
> same token and give it lots of compute, and similarly all experts can ignore a
> token so it is skipped for that layer. As well as improving training
> efficiency, this approach also has the benefits that it helps with load
> balancing and eliminates the need for auxiliary loss functions.

🌟 **Task Level MoEs, Various (2022)**
[DeMix pdf](https://arxiv.org/pdf/2108.05036.pdf),
[Task-MoE pdf](https://arxiv.org/pdf/2110.03742.pdf)

> Instead of routing each token separately these approaches use the same Expert
> for entire documents based on the task (which is supplied to the network).
> Instead of learning the routing, we supply the routing based on what we know
> about the tasks, inducing our own inductive bias. Also note that this offers
> memory footprint benefits at inference time - if inference is for a limited
> set of tasks, we only need these enough GPU memory for these experts.
> [ELMForest - Branch, Train, Merge (BTM)](https://arxiv.org/pdf/2208.03306.pdf%7D)
> and [c-BTM](https://arxiv.org/pdf/2303.14177.pdf) are follow-ups which use
> ensembling approaches from multiple LMs trained independently in a continual
> learning approach [code](https://github.com/kernelmachine/cbtm)

<!-- There are possibly additional benefits to combining task and token level experts. We could input a task and use this to decide which routers that we want to use in our MoE layers see [Multi-gate](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007).
Alternatively we could concatenate task information to the input of the router network so it can use it if it wants.
It's not clear whether having inductive bias would be better.  -->

**No Language Left Behind, Meta (2022)**
[pdf](https://arxiv.org/abs/2207.04672),
[official PyTorch code](https://github.com/facebookresearch/fairseq/tree/nllb)

> Translation is a natural setting for MoEs - some but not all of the parameters
> for English to Chinese translation be relevant in English to French
> translation as well. But using all of the English to Chinese knowledge might
> confuse the model. MoE therefore has useful inductive biases to allow this
> model to use only the relevant parts. Here the researchers show that the MoE
> approach scales even for extremely low-resource languages. Translation may be
> a natural environment for task/document-level rather than token-level routing.

**Hash Routing, Meta: Roller et al (2021)**
[pdf](https://browse.arxiv.org/pdf/2106.04426.pdf)

> Uses a static rather than fixed routing per input token and shows similar
> results to more principled routing methods in some regimes. Suggests that
> previous routing methods may be somewhat under-optimised.

**Switch Transformers, Google: Fedus et al (2021)**
[pdf](https://arxiv.org/pdf/2101.03961.pdf),
[review paper](https://arxiv.org/pdf/2209.01667.pdf),
[PyTorch code](https://nn.labml.ai/transformers/switch/index.html),
[model](https://huggingface.co/docs/transformers/model_doc/switch_transformers)

> Simplifies the MoE routing algorithm with top-1 routing. Shows that we can
> exploit the scaling laws with parameters as well as simply compute and
> develops distributed systems approach to MoE

**WideNet (Go Wider Instead of Deeper), NUS: Xue et al (2021)**
[pdf](https://arxiv.org/pdf/2107.11817.pdf)

> Suggests a parameter sharing approach using a single layer of multiple MoEs
> repeated multiple times as transformer blocks (similar to the Universal
> Transformer but with MoEs). This results in a deep model which has
> O(expert_num) instead of O(layer_depth) parameters. They achieve SoTA results
> with fewer parameters than previous models. More recently, Apple's
> [One Wide Feedforward paper](https://arxiv.org/pdf/2309.01826.pdf) details the
> amount of redundancy across layers. This suggests this approach is
> increasingly fruitful for on-device models.

🌟 **Outrageously Large Neural Networks (aka The Sparse MoE Layer), Google:
Shazeer et al (2017)** [pdf](https://arxiv.org/pdf/1701.06538.pdf)

> Introduces Mixture of Expert models in their modern form using Sparsely Gated
> MoE layer and a trainable gating network. They use RNNs as this is pre
> "Transformers Eating The World".

<!-- A Noam Shazeer, Geoff Hinton collab - two true legends of Deep Learning -->

## Other Modular Architectures

**Fast FeedForward (FFF), ETH Zurich: Belcak et al (2023)**
[pdf1](https://arxiv.org/pdf/2308.14711.pdf),
[pdf2](https://arxiv.org/pdf/2311.10770.pdf),
[official pytorch code1](https://github.com/pbelcak/fastfeedforward),
[official pytorch code2](https://github.com/pbelcak/UltraFastBERT),
[pytorch code](https://github.com/sap-ient-ai/FFF)

> Instead of the usual FeedForward Network, the authors propose a balanced tree
> structure where depending on your path through the tree, a different function
> is applied to the input. Inputs go either left or right through the tree
> depending on the result of a dot product with a learned discriminating vector.
> This approach results in encouraging performance with somewhat limited
> inference FLOPs but high training FLOPs and unstructured inference sparsity
> which must be applied sequentially which falls foul of the
> [Hardware Lottery](https://arxiv.org/pdf/2009.06489.pdf) and doesn't
> parallelise nicely on GPUs.

🌟 **Soft-MoE, Google DeepMind: Puigcerver et al (2023)**
[pdf](https://arxiv.org/pdf/2308.00951.pdf),
[pytorch code](https://github.com/fkodom/soft-mixture-of-experts/blob/main/soft_mixture_of_experts/soft_moe.py)

> Instead of Sparse MoE models, they allow each expert to using its router to
> select weights for a weighted average of input tokens that it wants to
> process. They show SoTA results on Image recognition tasks. Note since the
> approach relies on Expert Choice, it doesn't yet generalise to autoregressive
> generation.

## Early Exit: End-to-End Adaptive Computation

Early Exit approaches ask if we get the output of a neural network without going
through all the layers, particularly if faced with an easier example. This is
typically done by learning an exit probability at each layer.

**Sparse Universal Transformer (SUT), MILA: Tan et al (2023)**
[pdf](https://arxiv.org/pdf/2310.07096.pdf)

> Combines the Universal Transformer approach (RNN with transformer blocks) with
> the Mixture of Experts paradigm (multiple experts instead of a single FFN
> layer). They also use a new stick-breaking-based dynamic halting mechanism.
> This brings all the benefits of Sparse MoEs (such as less inference compute
> whilst having a lot of parameters) and the benefits of Universal Transformer
> (parameter efficiency, Turing-completeness and generalization ability)
> together.

**AdaTape, Google: Xue et al (2023)**
[pdf](https://arxiv.org/pdf/2301.13195.pdf),
[blog](https://ai.googleblog.com/2023/08/adatape-foundation-model-with-adaptive.html),
[official jax code](https://github.com/google-research/scenic/blob/main/scenic/projects/adatape/adatape_vit/adatape_vit.py)

> Extends the ACT method by giving the model a "tape" which contains some
> additional inputs which may be useful for encoding. For each token the model
> may append a variable number of tape tokens to the input, which allows it to
> regulate how much additional compute we add. The paper shows impressive
> performs on image classification tasks and the 'parity' task on long
> sequences.

**Dataset Pruning Using Early Exit Networks, Görmez et al (2023)**
[pdf](https://openreview.net/pdf?id=Kh114370zL)

> Early Exit Networks naturally learn which input examples are "easy" (can be
> exited early) or "difficult" (require all the layers of the network). The
> authors use this property to prune datasets to use for training and
> finetuning. The algorithm EEPrune achieves SOTA performance for dataset
> pruning in some regimes.

🌟 **PonderNet, DeepMind: Banino et al (2021)**
[pdf](https://arxiv.org/pdf/2107.05407.pdf),
[PyTorch code](https://github.com/koayon/ml-replications/tree/main/ponder)

> Allows the model to exit after each transformer layer if it's confident in the
> answer. It introduces a stable probabilistic policy for halting which provides
> low-variance unbiased gradient updates. This can also be combined with the
> [SkipNet](<[pdf](https://arxiv.org/pdf/1711.09485)>) paradigm where we instead
> of exiting directly, skip to the final few layers to allow our universal
> computation (applied to all inputs) to be at the end as well as the start of
> the network.

**PaBEE, DeepMind: Zhou et al (2020)**
[pdf](https://arxiv.org/pdf/2006.04152.pdf),
[official PyTorch code](https://github.com/huggingface/transformers/tree/main/examples/research_projects/bert-loses-patience)

> Introduces Patient Early Stopping. Whilst ACT has a learned exit probability,
> PABEE instead looks at the output class if it _were_ to exit. We exit if the
> intermediate outputs are the same over multiple layers. Interestingly they
> suggest that the reason for this isn't just speed; they suggest that early
> stopping will _improve_ performance due to lower risk of "overthinking"
> (analogously to stopping training earlier to prevent overfitting).
> [F-PaBEE](https://arxiv.org/pdf/2305.11916.pdf) prevents a slightly more
> flexible approach based on similarity scores.

**Universal Transformer, Google: Dehghani et al (2019)**
[pdf](https://arxiv.org/pdf/1807.03819.pdf)

> Reuses the transformer block recurrently across multiple layers with an
> ACT-like halting mechanism. RNNs can be better than transformers at length
> extrapolation but here we get the best of the Transformer (training
> parallelizability) and the best of the RNN (recurrent inductive bias).
> Universal Transformers can also be shown to be Turing-complete.

**Adaptive Computation Time (ACT) for RNNs, Google: Graves (2016)**
[pdf](https://arxiv.org/pdf/1603.08983.pdf)

> Introduces the ACT approach for models to learn how many computational steps
> they should take before returning an output. This approach is built on and
> refined in many later papers such as PonderNet.

## Adaptive Computation for Black-box models

For black box pre-trained models, perhaps those behind an API, there are some
techniques for using Adaptive Computation. These are promising techniques for
those with limited compute budgets.

Prompting techniques such as
[Reflexion](https://browse.arxiv.org/pdf/2303.11366.pdf),
[Debate](https://browse.arxiv.org/pdf/2305.14325.pdf),
[Chain of Thought](https://browse.arxiv.org/pdf/2201.11903.pdf),
[Tree of Thought](https://browse.arxiv.org/pdf/2305.10601.pdf) and
[Chain of Verification](https://browse.arxiv.org/pdf/2309.11495.pdf) can also be
used to improve performance for Black-box models.

🌟 **Contrastive Decoding, Stanford: Li et al (2022)**
[pdf](https://arxiv.org/pdf/2210.15097.pdf),
[pdf2](https://arxiv.org/pdf/2309.09117.pdf)

> A small helper model generates tokens alongside the main model. Tokens are
> up-weighted if the large model finds them proportionally much more plausible
> than the small model. This approach improves the quality of open-ended
> generations and reasoning ability. To extend this method towards additionally
> adaptive computation, smaller contrastive models could be applied
> conditionally depending on the input.

🌟 **Speculative Sampling, DeepMind: Chen et al (2023)**
[pdf](https://arxiv.org/pdf/2302.01318.pdf),
[pdf2](https://arxiv.org/pdf/2211.17192.pdf),
[blog](https://jaykmody.com/blog/speculative-sampling/),
[PyTorch code](https://github.com/jaymody/speculative-sampling)

> A smaller model generates multiple tokens autoregressively and then a larger
> model checks the smaller model against what it would have generated (all in
> one go). We accept only the tokens where the two models agree (by some
> acceptance criteria) and then the larger model's next token. This gives
> exactly the same output as the larger model would have but with significantly
> reduced sampling time. This takes advantage of the fact that we can
> parallelise evaluation whilst generation happens token by token. Additionally
> [Online Speculative Decoding](https://arxiv.org/pdf/2310.07177.pdf) suggests
> we can use any excess compute (at inference time) to retrain the small model
> online on the query distribution with teacher-student distillation.

<!-- The general principle here is that it's easier to evaluate than to generate. -->

**FrugalGPT, Stanford: Chen et al (2023)**
[pdf](https://arxiv.org/pdf/2305.05176.pdf)

> Details various approaches for fully black box adaptive computation (i.e. from
> an API where you don't even get logits). They use an LLM Cascade strategy
> where given a prompt they select n models to try sampling with, in order of
> increasing parameter count. The first model samples and we check the
> generation with a scoring function. If the generation is rejected then we
> generate with a more capable model. We continue this process until we accept
> the generation or are using the largest model. Interestingly this approach
> provides some shielding against
> [inverse scaling](https://arxiv.org/pdf/2306.09479.pdf) problems. They also
> use completion caching.

**Beam Search, Google: Sutskever et al (2014)**
[pdf](https://browse.arxiv.org/pdf/1409.3215.pdf)

> Beam search allows LMs to see the probability of choosing a few tokens at a
> time before selecting one by building out a tree. Increasing the number of
> beams increases the number of options explored downstream and hence the amount
> of compute per token.

<!-- Test-Time Compute is the System 2. We want to convert time to think into better outputs. We want to do structured reasoning over trees of possibilities. We want to convert time into better accuracy so when you think for longer, you get monotonically more accurate outputs.  -->

## Continual Learning

🌟 **Lifelong-MoE, Google DeepMind: Chen et al (2023)**
[pdf](https://arxiv.org/pdf/2305.12281.pdf)

> Trains a language model for multiple tasks by training for one task, freezing
> these weights and then adding some additional layers which can help to train
> the next task (in combination with the frozen layers) This treats pretrained
> weights more like an API (which you can use but not edit) when training a
> model to do a new task. This helps to eliminate the catastrophic forgetting
> that can happen with naive finetuning.

**Sparse Upcycling, Google Research: Komatsuzaki et al (2023)**
[pdf](https://arxiv.org/pdf/2212.05055.pdf)

> Shows that you can use an pre-trained dense model checkpoints as an
> initialisation for training sparse MoEs. This reduces the overall compute
> budget needed and reduces the sunk costs for already trained models. Sparse
> upcycling can be viewed as an efficient form of finetuning which converts a
> pretrained dense model to a sparse model for inference.

**MuNet, Google: Gesmundo et al (2022-23)**
[pdf](https://arxiv.org/pdf/2205.10937.pdf),
[pdf2](https://arxiv.org/pdf/2205.12755.pdf),
[pdf3](https://arxiv.org/pdf/2209.14745.pdf),
[pdf4](https://arxiv.org/pdf/2302.02721.pdf),
[official jax code](https://github.com/google-research/google-research/tree/master/muNet)

> Defines an evolutionary algorithm which adds different tasks onto an existing
> base model by (1) inserting adapter layers, (2) changing hyperparameters, (3)
> freezing layers and (4) copying layers to retrain. An interesting sketch of
> what Adaptive Computation could look like in the future.

## Tools & Agents

One way of varying compute is on some tokens calling out to an external API for
parts of completions.

🌟 **LLM-Powered Autonomous Agents, OpenAI: Lilian Weng (2023)**
[blog](https://lilianweng.github.io/posts/2023-06-23-agent/)

> An overview of agents as general problem-solvers powered by LLMs such as
> [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT) and
> [GPT-Engineer](https://github.com/AntonOsika/gpt-engineer) Agents typically
> can act within the world are are augmented with the ability to do explicit
> long term planning (via decomposing goals into sub-goals and learning from its
> mistakes), long-term memory (via a vector database) and tool use (calling
> external APIs).

**ChatGPT Plugins, OpenAI (2023)**
[blog](https://openai.com/blog/chatgpt-plugins),
[demo](https://chat.openai.com/?model=gpt-4)

> GPT-4 has access to plugins for tasks where it would be better suited to call
> an API. Examples include Code Interpreter, web browser and Wolfram Alpha.

<!-- RETRO, DeepMind:
> k-Nearest Neighbour approaches
 -->

**Toolformer, Meta: Schick et al (2023)**
[pdf](https://arxiv.org/pdf/2302.04761.pdf),
[pdf2](https://arxiv.org/pdf/2305.17126.pdf)

> Trained models to decide which APIs to call, when to call them, what arguments
> to pass, and how to best incorporate the results into future token prediction
> Effectively the LMs teach themselves how to use tools. In the limit case of
> this we simply require LMs/agents to be able to ask the right questions, know
> where to ask them and possibly be able to interpret the answers they receive.
> In other words, we offload the actual computation to external APIs (which may
> themselves be ML models) and use much smaller base models.

## Games

🌟 **Libratus: heads-up no-limit poker, Meta: Brown and Sandholm (2017)**
[pdf](https://www.science.org/doi/epdf/10.1126/science.aao1733),
[pdf2](https://arxiv.org/pdf/1705.02955.pdf),
[video](https://www.youtube.com/watch?v=2dX0lwaQRX0)

> The first AI to beat humans at Texas Hold Em Poker (heads up). An important
> part of the approach was in computing real-time responses to opponent moves,
> spending more compute on less obvious moves.

**AlphaGo/AlphaZero, DeepMind: Silver et al (2016)**
[pdf](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf),
[pdf2](https://www.nature.com/articles/nature24270.epdf),
[film](https://www.youtube.com/watch?v=WXuK6gekU1Y),
[blog](https://www.deepmind.com/research/highlighted-research/alphago)

> This result needs no introduction. In terms of Adaptive Computation, they the
> depth of the Monte Carlo Tree Search (MCTS) was allowed to be variable.

## Pre-cursors to Adaptive Computation

**Attention and The Transformer, Vaswani et al (2017)**
[pdf](https://arxiv.org/pdf/1706.03762.pdf)
[pdf2](https://arxiv.org/pdf/1409.0473.pdf)

> Although we don't normally think of it this way, attention can be viewed as a
> conditional computation mechanism. The matrix which is applied to the input is
> dependant on the incoming data.

**Conditional Computation, Bengio et al. (2016)**
[pdf](https://arxiv.org/pdf/1511.06297.pdf)

> They use Reinforcement Learning to train a policy gradient to decide which
> parts of the network to activate, in effect learning a dropout policy for
> sparsity.

**Adaptive Mixtures of Local Experts, Jacobs et al (1991)**
[pdf](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)

> Collaborative, learned Mixture of Experts approaches to handle subsets of the
> training set are proposed. It's remarkable how close current approaches are to
> the original gating network. They also show intuitive expert specialisation on
> the task of vowel discrimination.

## Open Source Libraries

**git_theta, UNC: Kandpal et al (2023)**
[pdf](https://arxiv.org/pdf/2306.04529.pdf),
[official framework-agnostic code](https://github.com/r-three/git-theta/)

> A git extension which allows tracking and merging changes to model checkpoints
> like git does with code. With git_theta you can see diffs in parameter groups
> and merge model finetuning branches with merging approaches. It's also
> efficient with low-rank changes to parameter groups.

🌟 **DeepSpeed-MoE, Microsoft: Rajbhandari et al (2022)**
[blog](https://www.microsoft.com/en-us/research/blog/deepspeed-advancing-moe-inference-and-training-to-power-next-generation-ai-scale/),
[pdf](https://arxiv.org/pdf/2201.05596.pdf),
[official PyTorch code](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/moe)

> Training and inference solution for distributed MoE models. They also present
> a new MoE architecture PR-MoE which has more experts in higher layers and a
> method for distilling expert models into dense 'student models'.

<!-- Sten (2022) [pdf](https://arxiv.org/pdf/2304.07613.pdf)
> PyTorch implementation of efficient, unstructured sparsity linear algebra operations with gradients.

-->

## AI Safety

With adaptive computation, models can choose to use more compute on harder
problems.

For problems where we're concerned about systems failing by not being able to do
sufficient computation then Adaptive Computation is very positive for Alignment.
We should expect fewer mistakes from a model utilising Adaptive Computation,
even on more difficult problems. Additionally,
[Adaptive Computation based systems are less susceptible to Adversarial Attacks](https://arxiv.org/pdf/2210.10253.pdf).
That is to say Adaptive Computation makes models more `robust`.

However, for problems where we're concerned about systems being deceptive or
mesa-optimising increasing the amount of inference-time compute increases their
ability to do so. Here the failure is not a "mistake" but entirely intentional
from the system's perspective. Inference-time search is one way that a model
could implement [deceptive alignment](https://arxiv.org/pdf/1906.01820.pdf) for
example.

## Other

**Sparse Scaling Laws, DeepMind: Frantar et al (2023)**
[pdf](https://arxiv.org/pdf/2309.08520.pdf)

> Scaling laws paper in the style of the
> [Chinchilla paper](https://arxiv.org/pdf/2203.15556.pdf). Details the optimal
> sparsity for a model given the inference FLOPs and training budget. They
> suggest that sparsity is especially important for larger models when seeing
> diminishing returns past Chinchilla optimality. See also
> [Unified Scaling Laws](https://browse.arxiv.org/pdf/2202.01169.pdf)

**FLOPs are all you need, Emin Orhan (2023)**
[blog](https://severelytheoretical.wordpress.com/2023/08/14/flops-are-all-you-need-a-conjecture-about-what-really-makes-deep-learning-work/)

> Short post detailing how the success of deep learning models is very
> correlated with the amount of compute that they use per parameter efficiently
> and how they share parameters.

**Review Paper: Dynamic Neural Networks Survey, Han et al (2022)**
[pdf](https://arxiv.org/pdf/2102.04906.pdf)

> A review of Adaptive Computation approaches.

**Scaling Scaling Laws with Board Games, Andy Jones (2021)**
[pdf](https://arxiv.org/pdf/2104.03113.pdf)

> [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)
> suggests that there are two general techniques that work well in Machine
> Learning - search and learning. This paper suggests that these can be traded
> off against one another - that is, instead of additional learning you could
> add capable search to achieve similar performance. We can trade off train-time
> and test-time compute depending on our requirements.

<!--

Lottery Tickets: if we prune we really do get sparsity but the problem is that the sparsity is not useful to us on modern hardware. We need block sparsity to take advantage of this. In the future it might be possible to use less structured sparsity and then this will become very relevant again.

-->

<!--
## Benchmarks

All the usual benchmarks apply but ones that are especially suited to Adaptive Computation methods include:

Parity

Complex logic questions

ContextQA location dataset

ARB (DuckAI benchmark)

Agents benchmarks

Sparsity May Cry (SMC)

-->

<!-- ## Approaches We're Excited To See Explored More

- When we have early exiting we essentially have to train classifiers for each layer in addition to the main model so we have additional overhead for training which is going to save us compute at inference tine. Are there better, more principled ways of early exiting at train time as well so that we don't have to learn very much from easy tokens?

- Current approaches to sparsity are mainly transformer with some sparsity added on the margin. Transformers have worked so well and people are generally leaving them alone and messing with everything else around them - we're interested in paradigm shift approaches which are completely sparse and move further away from the transformer.
-->

<br>

---

<br>

Thanks for reading, if you have any suggestions or corrections please submit a
pull request! And please hit the star button to show your appreciation.
