---
layout: post
title:  "Reading List"
date:   2016-06-20
categories: Reading List
---

# Neural Network Adaptation

* [Steve Renals ASRU 2015 Talk](http://homepages.inf.ed.ac.uk/srenals/asru2015-srenals.pdf)

 > Brief overview of challenges in NN adaptation
 
* [Sequence Summarizing Neural Network for Speaker
Adaptation](http://www.merl.com/publications/docs/TR2016-001.pdf)

 > **Abstract :**
In this paper, we propose a DNN adaptation technique, where the i-vector extractor is replaced
by a Sequence Summarizing Neural Network (SSNN). Similarly to i-vector extractor,
the SSNN produces a ”summary vector”, representing an acoustic summary of an utterance.
Such vector is then appended to the input of main network, while both networks are trained
together optimizing single loss function. Both the i-vector and SSNN speaker adaptation
methods are compared on AMI meeting data. The results show comparable performance of
both techniques on FBANK system with frame classification training. Moreover, appending
both the i-vector and ”summary vector” to the FBANK features leads to additional improvement
comparable to the performance of FMLLR adapted DNN system

* [SPEECH RECOGNITION WITH PREDICTION-ADAPTATION-CORRECTION RECURRENT
NEURAL NETWORKS](https://groups.csail.mit.edu/sls/publications/2015/YuZhang_ICASSP_2015.pdf)  

 > **Abstract :**  We propose the prediction-adaptation-correction RNN (PAC-RNN),
in which a correction DNN estimates the state posterior probability
based on both the current frame and the prediction made on the
past frames by a prediction DNN. The result from the main DNN is
fed back to the prediction DNN to make better predictions for the
future frames. In the PAC-RNN, we can consider that, given the
new, current frame information, the main DNN makes a correction
on the prediction made by the prediction DNN. Alternatively, it can
be viewed as adapting the main DNN’s behavior based on the prediction
DNN’s prediction. Experiments on the TIMIT phone recognition
task indicate that the PAC-RNN outperforms DNN, RNN, and
LSTM with 2.4%, 2.1%, and 1.9% absolute phone accuracy improvement,
respectively. We found that incorporating the prediction
objective and including the recurrent loop are both important to
boost the performance of the PAC-RNN.

* [DIFFERENTIABLE POOLING FOR UNSUPERVISED SPEAKER ADAPTATION](http://www.cstr.ed.ac.uk/downloads/publications/2015/Swietojanski_ICASSP2015.pdf)  

 > **Abstract :**  This paper proposes a differentiable pooling mechanism to perform
model-based neural network speaker adaptation. The proposed technique
learns a speaker-dependent combination of activations within
pools of hidden units, was shown to work well unsupervised, and
does not require speaker-adaptive training. We have conducted a set
of experiments on the TED talks data, as used in the IWSLT evaluations.
Our results indicate that the approach can reduce word error
rates (WERs) on standard IWSLT test sets by about 5–11% relative
compared to speaker-independent systems and was found complementary
to the recently proposed learning hidden units contribution
(LHUC) approach, reducing WER by 6–13% relative. Both methods
were also found to work well when adapting with small amounts of
unsupervised data – 10 seconds is able to decrease the WER by 5%
relative compared to the baseline speaker independent system.

* [LEARNING HIDDEN UNIT CONTRIBUTIONS FOR
UNSUPERVISED SPEAKER ADAPTATION OF NEURAL NETWORK ACOUSTIC MODELS](http://www.cstr.ed.ac.uk/downloads/publications/2014/ps-slt14.pdf)  

 > **Abstract :**  This paper proposes a simple yet effective model-based neural
network speaker adaptation technique that learns speakerspecific
hidden unit contributions given adaptation data,
without requiring any form of speaker-adaptive training, or
labelled adaptation data. An additional amplitude parameter
is defined for each hidden unit; the amplitude parameters
are tied for each speaker, and are learned using unsupervised
adaptation. We conducted experiments on the TED talks data,
as used in the International Workshop on Spoken Language
Translation (IWSLT) evaluations. Our results indicate that
the approach can reduce word error rates on standard IWSLT
test sets by about 8–15% relative compared to unadapted
systems, with a further reduction of 4–6% relative when
combined with feature-space maximum likelihood linear regression
(fMLLR). The approach can be employed in most
existing feed-forward neural network architectures, and we
report results using various hidden unit activation function.

* [On Speaker Adaptation of Long Short-Term Memory Recurrent Neural
Networks](https://www.cs.cmu.edu/~ymiao/pub/is2015_lstm.pdf)  

 > **Abstract :** Long Short-Term Memory (LSTM) is a recurrent neural network
(RNN) architecture specializing in modeling long-range
temporal dynamics. On acoustic modeling tasks, LSTM-RNNs
have shown better performance than DNNs and conventional
RNNs. In this paper, we conduct an extensive study on speaker
adaptation of LSTM-RNNs. Speaker adaptation helps to reduce
the mismatch between acoustic models and testing speakers.
We have two main goals for this study. First, on a benchmark
dataset, the existing DNN adaptation techniques are evaluated
on the adaptation of LSTM-RNNs. We observe that LSTMRNNs
can be effectively adapted by using speaker-adaptive
(SA) front-end, or by inserting speaker-dependent (SD) layers.
Second, we propose two adaptation approaches that implement
the SD-layer-insertion idea specifically for LSTM-RNNs. Using
these approaches, speaker adaptation improves word error
rates by 3-4% relative over a strong LSTM-RNN baseline. This
improvement is enlarged to 6-7% if we exploit SA features for
further adaptation.

* [DNN SPEAKER ADAPTATION USING PARAMETERISED SIGMOID AND RELU
HIDDEN ACTIVATION FUNCTIONS](http://mi.eng.cam.ac.uk/~cz277/doc/Conference-ICASSP2016-ADAPT.pdf)  

 > **Abstract :**   This paper investigates the use of parameterised sigmoid and recti-
fied linear unit (ReLU) hidden activation functions in deep neural
network (DNN) speaker adaptation. The sigmoid and ReLU parameterisation
schemes from a previous study for speaker independent
(SI) training are used. An adaptive linear factor associated with each
sigmoid or ReLU hidden unit is used to scale the unit output value
and create a speaker dependent (SD) model. Hence, DNN adaptation
becomes re-weighting the importance of different hidden units
for every speaker. This adaptation scheme is applied to both hybrid
DNN acoustic modelling and DNN-based bottleneck (BN) feature
extraction. Experiments using multi-genre British English television
broadcast data show that the technique is effective in both directly
adapting DNN acoustic models and the BN features, and combines
well with other DNN adaptation techniques. Reductions in word error
rate are consistently obtained using parameterised sigmoid and
ReLU activation function for multiple hidden layer adaptation.

# Research Groups
* [Microsoft Speech Research Group](https://www.microsoft.com/en-us/research/group/speech-dialog-research-group/)
 > Language model, Acoustic Model and Other NLP related areas are published here
* [Google Speech Research Group](http://research.google.com/pubs/SpeechProcessing.html)
 > Speech Recognition, TTS, NLP related arears

# Machine Learning
* [AI Principles, Stanford Course](http://web.stanford.edu/class/cs221/)

# Deep Learning
* [Deep NLP Stanford Course](http://cs224d.stanford.edu/syllabus.html)
* [Stanford Deep Learning Tutorial](http://ufldl.stanford.edu/tutorial/)
* [Stanford Convolution Neural Network](http://cs231n.github.io/)


# Word Embedding
* [Deep Learning, NLP and Representation](https://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
* [Word Embedding](http://sebastianruder.com/word-embeddings-1/)
* [Word Embedding](http://veredshwartz.blogspot.in/2016/01/representing-words.html)
* [Python Code for simple experiments](http://developers.lyst.com/2014/11/11/word-embeddings-for-fashion/)
* [Tomas mikalov Presentation](NIPS-DeepLearningWorkshop-NNforText.pdf)

> [Important characteristic of a word is the company it keeps](http://veredshwartz.blogspot.in/2016/01/representing-words.html). 
  According to the distributional hypothesis, words that occur 
  in similar contexts (with the same neighboring words), tend to 
  have similar meanings (e.g. elevator and lift will both appear 
  next to down, up, building, floor, and stairs); simply put, 
  "tell me who your friends are and I will tell you who you are" - the words version. 

# Neural Network Tips and Tricks
* [Deep NLP, Stanford](http://cs224d.stanford.edu/lectures/CS224d-Lecture6.pdf)

# Sequence to Sequence Learning and Neural Network with Attention

* [Sequence to Sequence Learning, Quoc, Le, Google](http://cs224d.stanford.edu/lectures/CS224d-Lecture16.pdf)
* [Machine transalation, part1](https://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-with-gpus/)
* [Machine transalation, part2](https://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-gpus-part-2/)
* [Machine transalation with attention, part3](https://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-gpus-part-3/)

# Dynamic Memory Networks

* [Ask Me Anything:Dynamic Memory Networks for Natural Language Processing](http://arxiv.org/pdf/1506.07285v5.pdf)
* [DMN for QnA, Metamind](http://cs224d.stanford.edu/lectures/CS224d-Lecture17.pdf)

# RNN for Slot filling

* [Using Recurrent Neural Networks for Slot Filling in Spoken Language Understanding](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/TASLP_RNN_SLU_2015.pdf)
* [Theano source code](https://github.com/mesnilgr/is13)

# Keyword Spotting Using NN

* [QUERY-BY-EXAMPLE KEYWORD SPOTTING USING LONG SHORT-TERM MEMORY NETWORKS](http://www.clsp.jhu.edu/~guoguo/papers/icassp2015_myhotword.pdf)
* [SMALL-FOOTPRINT KEYWORD SPOTTING USING DEEP NEURAL NETWORKS](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42537.pdf)
* [Online Keyword Spotting with a Character-Level Recurrent Neural Network](https://arxiv.org/pdf/1512.08903.pdf)
* [An application of recurrent neural networks to discriminative keyword spotting](http://www.cs.toronto.edu/~graves/icann_santi_2007.pdf)
* [AUTOMATIC GAIN CONTROL AND MULTI-STYLE TRAINING FOR ROBUST SMALL-FOOTPRINT KEYWORD SPOTTING WITH DEEP NEURAL NETWORKS](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43289.pdf)
* [Convolutional Neural Networks for Small-Footprint Keyword Spotting](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43969.pdf)


# Decoder Graph

* [Rapid Vocabulary Addition to Context-Dependent Decoder Graphs](http://www.isca-speech.org/archive/interspeech_2015/papers/i15_2112.pdf)
* [Discussion about graph](https://github.com/kaldi-asr/kaldi/issues/720)

# Language Model

* [Geo-location for Voice Search Language Modeling](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43817.pdf)
* [Word-Phrase-Entity Language Models: Getting More Mileage out of N-grams](https://www.microsoft.com/en-us/research/publication/word-phrase-entity-language-models-getting-more-mileage-out-of-n-grams/)  
  > We present a modification of the traditional n-gram language modeling approach that departs from the word-level data representation and seeks to re-express the training text in terms of tokens that could be either words, common phrases or instances of one or several classes. Our iterative optimization algorithm considers alternative parses of the corpus in terms of these tokens, re-estimates token n-gram probabilities and also updates within-class distributions. In this paper, we focus on the cold start approach that only assumes availability of the word-level training corpus, as well as a number of generic class definitions. Applied to the calendar scenario in the personal assistant domain, our approach reduces word error rates by more than 13% relative to the word-only n-gram language models. Only a small fraction of these improvements can be ascribed to a larger vocabulary.
* [Personalization of Word-Phrase-Entity Language Models](https://www.microsoft.com/en-us/research/publication/personalization-of-word-phrase-entity-language-models/)  
 > We continue our investigations of Word-Phrase-Entity (WPE) Language Models that unify words, phrases and classes, such as named entities, into a single probabilistic framework for the purpose of language modeling. In the present study we show how WPE LMs can be adapted to work in a personalized scenario where class definitions change from user to user or even from utterance to utterance. Compared to traditional classbased LMs in various conditions, WPE LMs exhibited comparable or better modeling potential without requiring pre-tagged training material. We also significantly scaled the experimental setup by widening the target domain, amplifying the amount of training material and increasing the number of classes.
* [Token-level Interpolation for Class-Based Language Models](https://www.microsoft.com/en-us/research/publication/token-level-interpolation-for-class-based-language-models/)
* [Rapidly building domain-specific entity-centric language models using semantic web knowledge resources](https://www.microsoft.com/en-us/research/publication/rapidly-building-domain-specific-entity-centric-language-models-using-semantic-web-knowledge-resources/)

# Acoustic Model

* [Speaker-aware Training of LSTM-RNNS for Acoustic Modelling](https://www.microsoft.com/en-us/research/publication/speaker-aware-training-of-lstm-rnns-for-acoustic-modelling/)

# Confidence Score/Rejection

* [Garbage Modeling for On-device Speech Recognition](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43847.pdf)

# Data Correction/Cleaning, Pronunciation Verification

* [FIX IT WHERE IT FAILS:PRONUNCIATION LEARNING BY MINING ERROR CORRECTIONS FROM SPEECH LOGS](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43263.pdf)


# Text Normalization

* [Shared Tasks of the 2015 Workshop on Noisy User-generated Text:Twitter Lexical Normalization and Named Entity Recognition](https://www.microsoft.com/en-us/research/wp-content/uploads/2015/08/WNUT19.pdf)
 * This paper has link to other papers and also link to some source code

# Learning Resources

* [Recent Deep Learning resources](http://handong1587.github.io/deep_learning/2016/07/08/keep-up-with-new-trends.html)
* [Deep Learning Resoruces](http://handong1587.github.io/categories.html#deep_learning-ref)

# RNN, LSTM

* [RNN, LSTM Resources](http://handong1587.github.io/deep_learning/2015/10/09/rnn-and-lstm.html)

## June, 2016

* [Smart Reply, Google Research Blog](https://research.googleblog.com/2015/11/computer-respond-to-this-email.html)
* [Thought Vector, Geoffrrey Hinton](http://www.extremetech.com/extreme/206521-thought-vectors-could-revolutionize-artificial-intelligence)

## July, 2016

* [Learning Compact Recurrent Neural Networks](http://arxiv.org/pdf/1604.02594v1.pdf)
* [Listen, Attend and Spell: A Neural Network for Large Vocabulary Conversational Speech Recognition](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/44926.pdf)


# G2P, Lexicon, Pronunciation Learning

## July, 2016

* [Learning Personalized Pronunciations for Contact Names Recognition](http://research.google.com/pubs/pub45415.html)
* [Automatic Pronunciation Verification for Speech Recognition](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43262.pdf)



# Speech Synthesis

## July, 2016

* [Acoustic Modeling for Speech Synthesis, Heiga, Zen](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/44630.pdf)


# Papers from Google Brain Team

Papers from [`Google Brain`](http://research.google.com/pubs/BrainTeam.html)

