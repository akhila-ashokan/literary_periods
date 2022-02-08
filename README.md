# Literary Periods

## Project Tracker 

Track the progress of this project [on Notion](https://www.notion.so/akhila-ashokan/Thesis-Planning-8f5a379bb79c4e1bb4c50346461a41c8).

## How to set up 

1) Request access to [HAL](https://wiki.ncsa.illinois.edu/display/ISL20/HAL+cluster) or another univerisity cluster.

2) Once access has been granted, download Git repository into home directory.

3) Set up a conda environment using the provided requirements.txt file. 

4) Install wget, download the gutenberg dammit zip file [here](https://github.com/aparrish/gutenberg-dammit) and save to data directory. 

5) You are done and ready to run the pipeline!

## How to run

### Scripts 
The majority of the pipeline relies on bash scripts included under the bash_script directory. Each literary period has it's own bash script for each python script. For example, Corpus.py would be run on the 1700s dataset by running `swbatch corpus_period.swb` on the cluster command line or setting up a cluster job composer to run the same bash script. The bash script are recommended here because most of these scripts should be run as batch processes that can take long periods of computation time and resources and that's the recommended run method by HAL. If time and resources are not an issue, the python scripts can be run directly from the command line. 

### Notebooks 
We use notebooks to run and visualize our python code. We have a notebook for data analysis that should be run at the very very beginning to unzip and save all the documents into the data directory. This same notebook can then be run to see stats about the corpus and to seperate the corpus by literary periods. The second notebook is a simple visualization notebook that produces plots. This should be run after Corpus.py and Model.py are run. 

#### Corpus.py 
This python script handles the majority of the corpus creation and manipulation. Each of the steps are broken down into arguments in this script. This script handles everything from tokenize to saving random contexts and sentences. 

#### Model.py
Once the context windows and descriptors have been extracted from each corpus, this script can be used to train the word2vec model. This script also run the PCA and creates PCA plots for later analysis. 


## Overview of Project 

###  Data Collection 

For this study, we utilized the *Gutenberg, dammit* corpus published by Allison Parrish and consisting of plain text Project Gutenberg files collected until June 2016. The corpus comes packaged into multiple sub-directories and the individual manuscripts are identified with unique, numerical Gutenberg IDs. The corpus also provides official Project Gutenberg metadata on each text in a JSON file. The entire corpus contains 50,729 manuscripts written by 18,462 unique authors. We narrowed down the corpus to include only books with ‘Fiction’ or ‘fiction’ in the genre and with ‘English’ as the only language within the text. If multiple genres were included, we selected manuscripts with at least one fiction genre. The metadata did not provide information on publication date so we determined the approximate publication century by examining the author’s birth year. We required that a valid author birth be given to at least one author so that we would assign each manuscript to a century. If a text had multiple authors born in different centuries, we used the first listed author’s birth date to determine the text’s publication century. Texts that were written prior to the 1500s were binned together. The majority of texts fell in the 1700s, 1800s, and 1900s bins so we selected these three centuries for further analysis. Figure 1 shows the distribution of the corpus based on the author's birth century. 


 With these requirements, we selected 7,487 manuscripts written by 1,956 unique authors for further analysis. The top 20 authors and top 20 genres in the filtered corpus are shown in Table 1. The filtered corpus, with the labeled author birth century, was tokenized and part of speech tagged using the en\_core\_web\_sm pipeline provided in the spacy library. In order to compare and contrast the literary periods, we break down our final corpus by the three major author birth centuries (i.e. 1700, 1800, 1900) into three separate corpuses, one for each literary period. For the reminder of the steps in our pipeline, we perform the same set of steps on each corpus in addition to the the combined corpus.


### Seed Words 

In order to extract descriptors relevant to our five sensory modalities, we applied a semi-automatic approach to select one set of non-overlapping seed words that are semantically and morphologically linked to each of the five sensory modalities. We started with a set of seed words manually selected for each sensory modality: sight (see/look), hear (hear/listen/sound), touch (touch/feel), taste (taste/flavor/savor), smell (smell/scent/odor). Then, we branched out from these base seed words to semantically related words using the WordNet lexical database. The WordNet lexical database is freely accessible as part of the nltk library. We used the WordNet database to automatically identify related words like hyponyms and hypernyms. We also removed any phrases that were linked to the base seed word. We utilized Python's word form library to automatically find morphological variations of the words. The final seed words list included 873 seed words across all modalities with 150-200 words per modality.  Each of the seed words was then part of speech tagged using the en\_core\_web\_sm pipeline provided in the spacy library. The same set of seed words was used to extract descriptors from all four corpuses. 

### Extracting Descriptors and Context Windows 

To extract the sensory descriptors from each corpus, we extracted content words that surround a seed word. Specifically, we looked at a context window of +- 4 centered around a seed word. Descriptors were only selected if they occurred at least n times within the context windows. We mainly opted to use a threshold of n to reduce the computation complexity and improve the feasibility of further calculations, but the threshold also helped to prune words that occurred less frequently within sensory context windows. The value of n was adjusted by manually examining the frequency of descriptors in each corpus. In addition, descriptors words were only selected if part of speech tagged as either a noun, verb, adjective, or adverb.  Context windows were also truncated if a sentence boundary was encountered. The seed words used to center context windows were not considered part of the context window but they were considered as descriptors if they occurred a sufficient amount of times in context windows. We selected a window size of 4 and a cut off threshold of 30 based on the parameters set in previous works, but the parameters should be further tested and validated for the data set. 

### Calculating the Perception Association Index 

We explored the capabilities of natural language processing presented in other works and applied similar techniques to eliminate noise in our descriptor list. After extracting descriptors from the corpus using the context windows, we narrowed down the list of descriptors by calculating the Perception Association Index (PAI) for each descriptor and selecting the top  300 sensory descriptors with the highest PAI value. The term Perception Association Index was originally mentioned in a study done by Girju et. al. PAI is an extension the Olfactory Association Index (OAI), introduced in another study conducted by Iatropoulos and colleagues on the semantic content of olfactory words. OAI measures how strongly a word is associated with the idea of smell. We note that this metric was validated on psychophysical datasets and literature has shown that high OAI ideas are linked to high rankings of olfactory association. PAI, which applies OAI to the other four senses as well, is the log2 probability that a descriptor d occurs in the in a sensory context rather than a non-sensory context. PAI allowed us to select a descriptors with a higher correlation to sensory semantics and visual them more easily on PCA. 

### Computation Model 

Using to our advantage the presence of sensory specific context words around our chosen seed words, we identified descriptors that express sensory content. We were then able to map out their semantic organization  by training word embedding models on the extracted context windows. Our word embedding model represents each word as a separate vector distance in a multi-dimensional space. This means words with a similar semantics will have similar vector representations. We trained a word2vec with CBOW model on the contexts windows using a hidden layer with 200 units, minimum word count as 1, and 30 training iterations. The model was trained on the set of all context windows extracted from each corpus with no distinction between senses. The distance between two descriptors, i and j, was calculated using p, the Pearson correlation between word vectors, as shown below:

Following this, the distance D between descriptors is converted to 0-1 range, with 0 indicating semantic identity and 1 indicating semantic opposition. Then, we calculate the distance matrix for each sensory modality, resulting in 5 matrices per corpus (20 matrices total). We visualize the descriptors for further analysis using Principal Component Analysis (PCA) with 2-components. For each corpus, we run PCA using the 5 distance matrices for each sensory modality. Additionally, we experimented with 3-component PCA using the same models to determine if there would be a better mapping of the sensory space. After manual inspection, we determined that a 3-component PCA was too complex for the size of the data sets to easily cluster descriptors upon examination. In the next section we examine the results from the PCA for each literary period corpus and explain the clustering of descriptors in more detail. 


### References 

[Inter-Sense: An Investigation of Sensory Blending in Fiction](https://arxiv.org/abs/2110.09710)

[Exploring the Sensory Spaces of English Perceptual Verbs in Natural Language Data](https://arxiv.org/abs/2110.09721)




