# cpp_deep

## Work distribution
### Beck
Use fg


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


