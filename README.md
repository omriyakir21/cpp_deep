# cpp_deep

## Work distribution
### Beck
Use Setfit model to train esm2.
https://huggingface.co/docs/setfit/main/en/index
1. Train only the classifier.
2. Train esm2 weights as well.
SetFit is an efficient and prompt-free framework for few-shot fine-tuning of Sentence Transformers.
To my understanding, the training with setfit function is done by contrastive learning.

Here’s a general outline of how few-shot fine-tuning works:

1. Pre-training: Start with a model that has been pre-trained on a large, diverse dataset. This could be a language model like GPT or a vision model like ResNet.

2. Task Definition: Define the specific task you want the model to perform. This might be a classification task, a regression task, or something else.

3. Few-Shot Data: Prepare a small dataset with a few examples for each class or category. This dataset is used to fine-tune the pre-trained model.

4. Fine-Tuning: Train the model on the few-shot dataset. During this phase, the model adjusts its weights based on the new data while retaining the general knowledge it acquired during pre-training.

### Aya
Use LoRA algorithm to fine tune esm2.
LoRA (Low-Rank Adaptation) is a method for fine-tuning pre-trained models efficiently by focusing on adapting a subset of model parameters. It’s particularly useful for large models where full fine-tuning might be computationally expensive or impractical.
https://huggingface.co/docs/diffusers/en/training/lora

Paper that used LoRA for protein language models.
https://www.biorxiv.org/content/biorxiv/early/2024/06/07/2023.12.13.571462.full.pdf


### Navit
Fine tune ProGen2 in a way you choose.
Github repository with the pre-trained models:
https://github.com/salesforce/progen/tree/main/progen2

The models are autoregressive transformers with next-token prediction language modeling as the learning objective (like chat-gpt). 

Link to the paper:
https://arxiv.org/pdf/2206.13517

### Omri
1. Improve data set, either by getting an existing one or by curating one from scratch.
2. Create baselines.

## Datasets Construction

### Positive Samples
The positive dataset was constructed using **CPPsite 2.0**, a manually curated database of cell-penetrating peptides. CPPsite 2.0 provides comprehensive information on cell-penetrating peptides, including their sequences, structures, and biological activities.

### Unclassified Dataset
#### BIOPEP-UWMix
The unclassified dataset was sourced from **BIOPEP-UWMix**, a database of human bioactive peptides. BIOPEP-UWMix contains a diverse collection of peptides with various bioactivities, providing a rich resource for bioactive peptide research.

__TODO:__ Fetch the dataset from the database.
#### SPENCER\_Peptide\_info
SPENCER is a catalog of small peptides encoded by ncRNAs. The ncRNA encoding peptides were derived from re-annotation of public mass spectrometry data from over 1,700 patient samples spanning diverse cancers.


__TODO:__ Understand if the dataset is useful for the project.
#### PeptideAtlas
PeptideAtlas is a multi-species compendium of peptides identified in a large number of tandem mass spectrometry experiments. The PeptideAtlas database contains a wealth of information on peptides identified in various biological samples.

__Issue__:
Peptide atlas: It is not clear whether PeptideAtlas is relevant because most of these peptides do not have a biological function.

The way they get this data is:
1. Take sample (e.g. from blood).
2. Use proteases at high concentration to cleave all the consitutitve proteins into peptides.
3. Identify the peptides using mass spectrometry

So, many of these peptides are just digested protein fragments… But some of them could be functional, it’s not clear.

__Issue__: The dataset is huge. If we are going to use it we will need representative samples.

## Embeddings
### esm2 
__Issue__: Creating the embedding we get this message:
"Some weights of EsmModel were not initialized from the model checkpoint at facebook/esm2_t6_8M_UR50D and are newly initialized: ['esm.pooler.dense.bias', 'esm.pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."


