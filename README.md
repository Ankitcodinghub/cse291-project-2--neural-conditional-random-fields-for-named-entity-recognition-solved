# cse291-project-2--neural-conditional-random-fields-for-named-entity-recognition-solved
**TO GET THIS SOLUTION VISIT:** [CSE291 Project 2- Neural Conditional Random Fields for Named Entity Recognition Solved](https://www.ankitcodinghub.com/product/cse-291-advanced-statistical-nlp-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;126779&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;3&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (3 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSE291 Project 2- Neural Conditional Random Fields for Named Entity Recognition Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (3 votes)    </div>
    </div>
1 Overview

In this assignment, you will be building a neural conditional random field (CRF) for the task of named entity recognition (NER). After introducing the task and modeling approach in Sections 2–4, we provide implementation instructions in Section 5. Finally, we describe deliverables in Section 6 and optional extensions to the project in Section 8.

2 Named Entity Recognition

Named Entity Recognition (NER) is a type of information extraction task with the goal of identifying and classifying word spans (sequences of consecutive words in a sentence) into categories such as person, organization, location, time, and quantity. NER is an important processing step used to extract keywords for applications such as structured knowledge base creation, dictionary building, search, and voice command understanding. The task is best illustrated with an example:

Andrew Viterbi co-founded Qualcomm, a company headquartered in San Diego. After labeling each named entity span with its category:

[Andrew Viterbi]person co-founded [Qualcomm]organization, a company headquartered in [San Diego]location.

Even though, NER is a span/chunk identification and classification task, it is typically set up as a word tagging problem by annotating tokenized named entities with class-labeled Beginning-InsideOutside (BIO) tags. In this format, words outside named entity chunks are labeled with an ‘O’ tag and words inside named entity chunks are labeled with their class label prefixed with ‘I’. For entities that are immediately next to each other, the first word of the second entity gets labeled as ‘B’. For example, the above, tokenized sentence under this labeling scheme looks like this:

Andrewi-per Viterbii-per co-foundedo Qualcommi-org ,o ao companyo headquarteredo ino Sani-loc Diegoi-loc .o

For this assignment, you will be using data from the CoNLL-2003 Shared Task , which only considers 4 classes of named entities: persons, locations, organizations and names of miscellaneous entities that do not belong to the previous three groups.

Figure 1: Neural network-based tagging models. On the left, a bidirectional RNN makes independent tag classifications per time step. On the right, a bidirectional RNN-CRF learns local sequential dependencies between neighboring output tags to predict the structured output tag sequence. Diagram based on [4].

3 Baseline: RNN for NER

As we’ve learned from lectures and the previous language modeling assignments, recurrent neural networks (RNNs) are a natural choice for computing learned feature representations of sequences.

Given a labeled sentence {X,Y } composed of a sequence of word embedding vectors x1 …xT and tag embedding vectors y1 …yT, RNNs apply a learned recurrent function using input weights/bias

Wih,bih and recurrent weights/bias Whh,bhh

−→

h t = tanh(Wihxt + bih + Whhh(t−1) + bhh). (1)

For bidirectional RNNs, hidden states −→h t,←−h t are computed in forward and backward directions, respectively, and concatenated to obtain ht = h−→h t;←−h ti.

In order to predict the associated tag for time step t, we apply a logistic regression classifier on input features ht using learned weights/bias Wout,bout and the softmax function to obtain a probability distribution over the tag set vocabulary.

ft = Woutht + bout (2) softmax(ft) (3)

We show a diagram of a bidirectional RNN for NER on the left of Figure 1. (NB: We defined the model with a vanilla RNN for illustration, but you can (and probably should) use an LSTM here instead of an RNN.)

Learning: Similar to the previous assignments, parameters can be learned by minimizing the sum of the negative log likelihood under the model parameters θ for each prediction at each time step over training set X,Y∗ using the ground truth tags at each time step denoted by y∗t:

X,Y∗ T

NLL (4)

In practice, we compute the loss stochastically over batches instead.

Decoding: We simply apply an argmax at each time step independently since each classification is performed independently.

4 Advanced: RNN-CRF for NER

Although the RNN tagger (Sec. 3) produces rich, context-sensitive feature representations of the input sentence, the independent tag classification decisions on top of these features are suboptimal for sequence labeling. Without any dependency between tags, the model is unable to learn the constraints that are common to sequence labeling tasks. For instance, in NER i-per does not directly follow i-org, and in English part-of-speech tagging, adjectives only precede nouns. To remedy this, we can use a Conditional Random Field (CRF) layer to incorporate dependency structure between neighboring labels.

The CRF, introduced in [3], is a class of globally normalized, discriminative model that estimates the probability of an entire sequence p(Y | X) in a log-linear fashion on feature potentials (i.e., nonnegative functions) Φ:

Φ(X,Y )

p(Y | X;θ) = P Y 0∈O Φ(X,Y 0), (5)

The denominator is also referred to as the “partition function’. The O in the denominator represents the set of all possible sequences of output tags over the input sentence. (NB: Strictly speaking, O should depend on the input sentence, X, since the length of the input determines the length of the tag sequence. We omit this dependence for notational brevity.) This can present a major computational burden unless we assume that Φ is composed of a bunch of local potential functions that only operate on the preceding local neighbor yt−1:

. (6)

Under this linear chain assumption, we can achieve tractable learning/decoding with dynamic programming algorithms.

Instead of manually defining our feature potentials, which takes considerable human effort, we can automatically learn nonlinear features in a deep neural network. To do this, we can use our transformed RNN input feature potentials (or “emissions”), ft, from Sec. 3 as our log emission potentials. Our log transition potentials can simply be a square weight matrix U over the tag set representing the weight of the transition from one tag to the next.

T

Φ(X,Y ) = Yφemission(X,yt) · φtransition(yt−1,yt) (7)

. (8)

We show a diagram of a CRF layer on top of bidirectional RNN emissions on the right of Figure 1. Learning: In order to estimate model parameters θ, we minimize the negative log likelihood over entire sequences in the dataset {X,Y∗}:

X,Y∗ X,Y∗

NLL = X −logp(Y ∗ | X;θ) = X −hlogΦ(X,Y ∗) − log X Φ(X,Y )i (9)

Y ∈O

Decoding: During decoding we aim to find the most likely tag sequence under the model parameters θ

arg maxY ∈Op(Y | X;θ), (10)

which can be computed exactly using the Viterbi algorithm (see pseudocode at Wikipedia ), which has a recurrence very similar to the Forward algorithm dynamic program.

5 Implementation Instructions

Implement a linear chain CRF using the bidirectional recurrent parameterization defined above and use it for the task of NER. You are encouraged to clone and use the starter code from the CSE291 repository , which includes portions of the CoNLL-2003 Shared Task NER data train/dev splits, data loading functionality, and evaluation using span-based precision, recall, and F1 score. Additionally, we provide a bidirectional LSTM tagger baseline BiLSTMTagger based on Section 3 that performs independent classifications over the tag set at each time step. We suggest that you add the required Forward algorithm for CRF negative log likelihood learning and the Viterbi algorithm for exact CRF decoding on top of the existing BiLSTMTagger class.

Using the provided evaluation procedure, we can visualize the chunk-level precision, recall, and F1 for each class of named entity. Here’s the best scoring evaluation result (72.42 avg Fβ=1) on the dev.data.quad for BiLSTMTagger after 30 epochs of training on train.data.quad (6 mins training time on Colab GPU, not vectorized or batched): processed 11170 tokens with 1231 phrases; found: 1180 phrases; correct: 873. accuracy: 75.00%; (non-O)

accuracy: 94.42%; precision: 73.98%; recall: 70.92%; FB1: 72.42

LOC: precision: 83.85%; recall: 81.54%; FB1: 82.68 353

MISC: precision: 80.14%; recall: 58.85%; FB1: 67.87 141 ORG: precision: 62.25%; recall: 61.24%; FB1: 61.74 302

PER: precision: 71.88%; recall: 74.80%; FB1: 73.31 384

Andhere’sthebestscoringevaluationresult(74.03avgFβ=1)onthedev.data.quadforourBiLSTMCRFTagger (20 mins training time on Colab GPU, vectorized but not batched): processed 11170 tokens with 1231 phrases; found: 1141 phrases; correct: 878. accuracy: 75.72%; (non-O)

accuracy: 94.56%; precision: 76.95%; recall: 71.32%; FB1: 74.03

LOC: precision: 87.65%; recall: 78.24%; FB1: 82.68 324

MISC: precision: 81.43%; recall: 59.38%; FB1: 68.67 140 ORG: precision: 68.86%; recall: 61.24%; FB1: 64.83 273

PER: precision: 72.28%; recall: 79.13%; FB1: 75.55 404

As a reminder, we will not be grading based on the performance of your method, but the quality of the presentation and analysis in your report write-up.

6 Deliverables

Please submit a 2–3 page report along with the source code on Gradescope. We suggest using the ACL style files, which are also available as an Overleaf template. In your submission, you should describe your implementation choices and report performance using the evaluation code in the appropriate graphs and tables. In addition, we expect you to include some of your own investigation or error analysis. We are more interested in knowing what observations you made about the models or data than having a reiteration of the formal definitions of the various models. If you choose to complete an optional task, please present an analysis of your findings, along with the particular implementation decisions you made.

7 Useful References

While you can use the following PyTorch tutorial as a reference, the code in your implementation mustbeyourown. http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html

8 Optional Extensions

• Choose a pre-trained language model (like BERT) and use its embeddings as input to the CRF model.

• Add local, hand-designed features to your neural CRF by creating feature functions aimed to encode important information for the task of NER, such as capitalization, word form, or the presence of a word in a gazetteer. Do these features help performance in low-resource training scenarios?

• Choose another sequence labeling task to apply your Neural CRF to. In your write-up, describe the task, the dataset, and the evaluation setup. Here are some suggested tasks:

– Part of speech tagging in a language of your choice. How does the performance of your method scale with tag set size?

– Co-reference resolution

– NP Chunking (i.e., shallow parsing)

– Word sense disambiguation

• Extend your linear chain CRF into a semi-Markov CRF for NER that jointly segments the text into the appropriate named entity phrase class (PER, LOC, etc.) and tags them with the appropriate I/O label.

• Choose your own adventure! Propose and implement your own analysis or extension.

You are allowed to discuss the assignment with other students and collaborate on developing algorithms – you’re even allowed to help debug each other’s code! However, every line of your write-up and the new code you develop must be written by your own hand.

References

[1] Michael Collins. Log-Linear Models, MEMMs, and CRFs.

[2] Zhiheng Huang, Wei Xu, and Kai Yu. Bidirectional lstm-crf models for sequence tagging. arXiv preprint arXiv:1508.01991. 2015.

[3] John Lafferty, Andrew McCallum, Fernando Pereira. Conditional random fields: Probabilistic models for segmenting and labeling sequence data. Proc. 18th International Conf. on Machine Learning. Morgan Kaufmann. pp. 282–289. 2001.

[6] Erik F. Sang and Fien De Meulder. Introduction to the CoNLL-2003 shared task: Languageindependent named entity recognition. CoNLL. 2003.
