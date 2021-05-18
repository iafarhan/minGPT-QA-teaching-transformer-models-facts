### Teaching Transformer Models facts about the world through pretraining, and accessing that knowledge through finetuning.


We train a Transformer model to attempt to answer simple questions of the form *Where was person born?* – **without providing any input text from which to draw the answer.** 
we found that models are able to learn some facts about where people were born through pretraining, and access that information during fine-tuning to answer the questions. 
This fact is inspired from [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

**Birth place** files contains pair of name and a place that particular place was born.
First we focus on the Finetuning without Pretraining. doing so gave pretty bad results on our transformer model on this this of extractive question answering. 
Using the above paper as motivation, we pretrained our model on a large wikipedia text. Details are as follow
- So, first we do Span corruption. We design an objective that randomly samples and then drops out 15% of tokens in the input sequence. All consecutive spans of dropped-out tokens are replaced by a single sentinel token. Each sentinel token is assigned a token ID that is unique to the sequence. [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf)
- We Pretrain our model on this span corruption task. The model is able to learn some interesting facts about people were born during this pretraining and Finetuning the model on birth place dataset gave amazing results.

We have implemented different variants of MultiheadAttention mechanism as well. 
- CausalSelfAttention
- SynthesizerSelfAttention

**Causal Self-Attention** is the vanilla multi-head masked Self-attention layer with a projection at the end. we have used scaled-dot product as our scoring function in this case.

**Synthesizer Self-Attention** is a very recent alternative to causal self-attention that has potential benefits by removing this dot product. In vanilla self-attention the scoring function returns a block_size * block_size attention scores. This computation is quaratic in the sequence's length. Synthesizer self-attention overcomes this and computes the block_size * block_size matrix of attention scores directly. it is inspired from [Synthesizer: Rethinking Self-Attention in Transformer Models](https://arxiv.org/abs/2005.00743)

Each model with different attention variants take approximately 2 hours to train. 
