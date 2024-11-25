# cpp_deep

## Abstract
Cell-penetrating peptides (CPPs) play a pivotal
role in drug delivery and therapeutic development, offering a means to transport molecules
across cellular membranes that are otherwise
impermeable. The accurate classification of
peptides as cell-penetrating or non-penetrating
is therefore critical for advancing research in
targeted therapies and biotechnology. This
project focuses on developing an efficient peptide classification system by fine-tuning the
ESM2 protein language model. We implemented four models for this task: a fewshot learning model using the SetFit framework, a fully fine-tuned ESM2 model with
an added classification head, an ESM2 model
fine-tuned with LoRA (Low-Rank Adaptation),
and a baseline convolutional model. To ensure reliable performance, we employed crossvalidation and incorporated class weights to
handle imbalanced data. Surprisingly, the
convolutional baseline model outperformed
all other approaches, followed by the LoRA
fine-tuned model, the fully fine-tuned ESM2
model, and the few-shot learning model. Our
dataset comprises curated CPPs from CPPsite 2.0 and bioactive peptides from BIOPEPUWMix. In training, we treated the BIOPEPUWMix dataset as our negative set, recognizing
that its peptides are not explicitly known to penetrate or not penetrate cells. From this dataset,
we identified top false positives as promising
new CPP candidates, highlighting the potential of our approach to expand the repertoire of
known cell-penetrating peptides. This scalable
and computationally efficient solution not only
distinguishes cell-penetrating peptides but also
provides a framework for discovering novel
CPPs, contributing to advancements in peptidebased therapeutic research.
