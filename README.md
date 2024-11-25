# cpp_deep

## Abstract
Cell-penetrating peptides (CPPs) play a pivotal
role in drug delivery and therapeutic develop￾ment, offering a means to transport molecules
across cellular membranes that are otherwise
impermeable. The accurate classification of
peptides as cell-penetrating or non-penetrating
is therefore critical for advancing research in
targeted therapies and biotechnology. This
project focuses on developing an efficient pep￾tide classification system by fine-tuning the
ESM2 protein language model. We imple￾mented four models for this task: a few￾shot learning model using the SetFit frame￾work, a fully fine-tuned ESM2 model with
an added classification head, an ESM2 model
fine-tuned with LoRA (Low-Rank Adaptation),
and a baseline convolutional model. To en￾sure reliable performance, we employed cross￾validation and incorporated class weights to
handle imbalanced data. Surprisingly, the
convolutional baseline model outperformed
all other approaches, followed by the LoRA
fine-tuned model, the fully fine-tuned ESM2
model, and the few-shot learning model. Our
dataset comprises curated CPPs from CPP￾site 2.0 and bioactive peptides from BIOPEP￾UWMix. In training, we treated the BIOPEP￾UWMix dataset as our negative set, recognizing
that its peptides are not explicitly known to pen￾etrate or not penetrate cells. From this dataset,
we identified top false positives as promising
new CPP candidates, highlighting the poten￾tial of our approach to expand the repertoire of
known cell-penetrating peptides. This scalable
and computationally efficient solution not only
distinguishes cell-penetrating peptides but also
provides a framework for discovering novel
CPPs, contributing to advancements in peptide￾based therapeutic research.