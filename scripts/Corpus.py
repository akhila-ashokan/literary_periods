import spacy, sys, time, argparse, re, math, faulthandler, logging
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import multiprocessing as mp
import numpy as np
from nltk.tokenize import sent_tokenize
from os.path import exists
import plotly.express as px
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer

## TODO: Add timing decorator 

nlp = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer", "textcat", "entity_linker", "entity_ruler", "textcat_multilabel", "morphologizer", "attribute_ruler", "transformer"])
nlp.disable_pipe("parser")
nlp.enable_pipe("senter")
stopwords = nlp.Defaults.stop_words
logging.basicConfig(level = logging.DEBUG)


class Corpus: 
    def __init__(self, args):
        self.corpus_df = pd.read_csv('../data/english_fiction_metadata.csv')
        if args.corpus_type == '1700':
            self.corpus_df = self.corpus_df.loc[(self.corpus_df['Author Birth Century'] == '1700')]
        elif args.corpus_type == '1800':
            self.corpus_df = self.corpus_df.loc[(self.corpus_df['Author Birth Century'] == '1800')]
        elif args.corpus_type == '1900':
            self.corpus_df = self.corpus_df.loc[(self.corpus_df['Author Birth Century'] == '1900')]
        logging.info(self.corpus_df.shape)
            
        self.modalities = ['sight', 'hear', 'touch', 'taste', 'smell']
        self.seed_words = self.read_file('../data/seed_words/seed_words.pickle')
        self.context_windows_path = '../data/' + args.corpus_type + '/context_windows.pickle'
        self.descriptors_path = '../data/' + args.corpus_type + '/descriptors.pickle'
        self.filtered_descriptors_path = '../data/' + args.corpus_type + '/filtered_descriptors.pickle'
        self.word_freq_path = '../data/' + args.corpus_type + '/word_freq.pickle'
        self.pai_df_path = '../data/' + args.corpus_type + '/pai_df.pickle'
        self.random_sentences_path = '../data/' + args.corpus_type + '/random_sentences.pickle'
        self.random_contexts_path = '../data/' + args.corpus_type + '/random_contexts.pickle'
        self.random_seed_words_path = '../data/' + args.corpus_type + '/random_seed_words.pickle'
        self.pai_scatterplot_path = '../visuals/' + args.corpus_type + '/' + args.corpus_type + '_pai_scatterplot.html'
        self.pai_histogram_path = '../visuals/' + args.corpus_type + '/' + args.corpus_type + '_pai_histogram.html'
        self.tf_idf_path = '../data/' + args.corpus_type + '/tf_idf_' + args.tf_idf_method + '.pickle'
        self.tf_idf_scatter_plot = '../visuals/' + args.corpus_type + '/' + args.corpus_type + '_tf_idf_scatterplot_' + args.tf_idf_method + '.html'
        self.nr_word_freq_path = '../data/' + args.corpus_type + '/' + args.corpus_type + '_nr_word_freq.pickle'
        self.context_windows = {'sight': [], 'hear': [], 'touch': [], 'taste': [], 'smell': []}
        self.descriptors = {'sight': {}, 'hear': {}, 'touch': {}, 'taste': {}, 'smell': {}}
        self.filtered_descriptors =  {'sight': {}, 'hear': {}, 'touch': {}, 'taste': {}, 'smell': {}}
        self.word_freq = {}
        self.pai_df = pd.DataFrame(columns = ['word', 'modality', 'total_freq', 'total_freq_in_sense', 'PAI'])
        self.random_sentences =  {'sight': {}, 'hear': {}, 'touch': {}, 'taste': {}, 'smell': {}}
        self.random_contexts = {'sight': {}, 'hear': {}, 'touch': {}, 'taste': {}, 'smell': {}}
        self.random_seed_words = {'sight': {}, 'hear': {}, 'touch': {}, 'taste': {}, 'smell': {}}
        self.tf_idf = pd.DataFrame(columns = ['word', 'modality', 'TF-IDF', 'word_freq'])
        self.nr_word_freq = {}
        
       
    def read_file(self, input_path):
        logging.info('Opened: ' + input_path)
        with open(input_path, "rb") as f:
            input_obj = pkl.load(f)
            return input_obj
    
    def save_file(self, output_obj, output_path):
        with open(output_path, 'wb') as f:
            pkl.dump(output_obj, f)
        logging.info('Saved to: ' + output_path)
        
    def tokenize_doc(self):
        vfunct = np.vectorize(self.tokenize)
        vfunct(self.corpus_df.original_path, self.corpus_df.tokenized_path) 
        
    def create_tokenized_file(self, sents, larger_dataset):
        sentences = []
        for sent in sents:
                sentence = []
                if larger_dataset:
                    sent = nlp(sent)
                for token in sent:
                    word = (str(token.text).lower(), str(token.pos_))
                    sentence.append(word)
                sentences.append(sentence)
        return sentences
            
    def tokenize(self, original_path, tokenized_path):
        if (not exists(tokenized_path)):
            start_time = time.time()
            logging.info('--Opening file: ' + original_path)
            with open(original_path) as f:
                text = f.read().replace('\n', ' ')
                if len(text) > 1000000:
                    logging.info('--Working with larger dataset--')
                    sents = sent_tokenize(text)
                    larger_dataset = True
                else:
                    nlp.max_length = len(text) + 10
                    sents = nlp(text).sents
                    larger_dataset = False
                sentences = self.create_tokenized_file(sents, larger_dataset) 
            self.save_file(sentences, tokenized_path)
            end_time = time.time()
            logging.info(end_time - start_time)
        
            
    def process_doc(self,):
        vfunct = np.vectorize(self.extract_descriptors)
        vfunct(self.corpus_df.tokenized_path, self.corpus_df.cw_df_path)  
            
    def extract_descriptors(self, tokenized_path, cw_df_path):
        # rerun with if statement to check if context exists or not 
        window_size = 4
        punct = ['.', ',', ':', ';', '"', '!', "?"]
        ok_pos = ['ADJ', 'ADV', 'NOUN', 'VERB']
        start_time = time.time()
        text = self.read_file(tokenized_path)
        doc_process_df = pd.DataFrame(columns = ['sense_name', 'seed_word', 'context_window', 'sentence'])

        for i, sent in enumerate(text):
            for j, word in enumerate(sent):
                if word in self.seed_words['seed_word'].tolist():
                    start = max(j - window_size, 0)
                    end = min(j + window_size + 1 , len(sent)) 
                    context = [sent[k] for k in range(start, end) if k != j and sent[k][0] not in stopwords and sent[k][0] not in punct and re.search("[A-Za-z]", sent[k][0]) and sent[k][1] in ok_pos]
                    doc_process_df = doc_process_df.append({'sense_name': self.seed_words.loc[self.seed_words['seed_word'] == word, 'sense_name'].iloc[0],
                                                            'seed_word' : word,
                                                            'context_window' : context,
                                                            'sentence': sent}, ignore_index = True)
        end_time = time.time()
        logging.info(end_time - start_time)
        self.save_file(doc_process_df, cw_df_path)
    
    def combine_context_windows(self,):
        vfunct = np.vectorize(self.add_to_context_windows)
        vfunct(self.corpus_df.cw_df_path)   
        self.save_file(self.context_windows, self.context_windows_path)
        
    def add_to_context_windows(self, cw_df_path):
        start_time = time.time()
        cw_df = self.read_file(cw_df_path)
        for modality in self.modalities: 
            text_context_windows = [row['context_window'] for index, row in cw_df.iterrows() if row['context_window'] and row['sense_name'] == modality]
            self.context_windows[modality] = self.context_windows[modality] + text_context_windows
        end_time = time.time()
        logging.info(end_time - start_time)
        
    def combine_desciptors(self,):
        vfunct = np.vectorize(self.add_to_descriptors)
        vfunct(self.corpus_df.cw_df_path)   
        self.save_file(self.descriptors, self.descriptors_path)
        
    def add_to_descriptors(self, cw_df_path):
        start_time = time.time()
        cw_df = self.read_file(cw_df_path)
        cw_df.apply(lambda row: self.count_descriptors(row), axis = 1)
        end_time = time.time()
        logging.info(end_time - start_time)
        
    def count_descriptors(self,row):
        for word in row['context_window']:
            if word in self.descriptors[row['sense_name']]:
                self.descriptors[row['sense_name']][word] += 1
            else:
                self.descriptors[row['sense_name']][word] = 1

                
    def apply_descriptor_threshold(self,threshold):
        start_time = time.time()
        descriptors = self.read_file(self.descriptors_path)
        counts = []
        
        for modality, descriptor_dict in descriptors.items():
            logging.info('Length of descriptor list for ' + modality + ': ' +  str(len(descriptors[modality])))
            logging.info(set(list(descriptor_dict.values())))
            for word, count in descriptor_dict.items():
                if count not in counts:
                    counts.append(count)
                if count >= threshold: 
                    self.filtered_descriptors[modality][word] = count
            logging.info('Length of filtered descriptor list for ' + modality + ': ' +  str(len(self.filtered_descriptors[modality])))

        self.save_file(self.filtered_descriptors, self.filtered_descriptors_path)    
        end_time = time.time()
        logging.info(end_time - start_time)
        
    def calculate_PAI(self,):
        start_time = time.time()
        filtered_descriptors = self.read_file(self.filtered_descriptors_path)
        word_freq = self.calculate_word_freq()
        
        for modality, descriptor_dict in filtered_descriptors.items():
            for word, count in descriptor_dict.items():
                pai = math.log2(count/word_freq[word])
                row = {'word': word, 'modality': modality, 'total_freq': word_freq[word], 'total_freq_in_sense': count,'PAI': pai}
                self.pai_df = self.pai_df.append(row, ignore_index = True)
        self.pai_df = self.pai_df.sort_values(by=['PAI'], ascending=False)
        self.save_file(self.pai_df, self.pai_df_path)
        end_time = time.time()
        logging.info(end_time - start_time)       
        
    def calculate_word_freq(self,):
        vfunct = np.vectorize(self.count_words)
        vfunct(self.corpus_df.tokenized_path)   
        self.save_file(self.word_freq, self.word_freq_path)
        return self.word_freq
        
    def count_words(self, tokenized_path):
        text = self.read_file(tokenized_path)
        for i, sent in enumerate(text):
            for j, word in enumerate(sent):
                if word in self.word_freq:
                    self.word_freq[word] += 1
                else:
                    self.word_freq[word] = 1

    def create_PAI_visuals(self,):
        pai = self.read_file(self.pai_df_path)
        fig = px.scatter(pai, 
                        x=pai["total_freq_in_sense"],
                        y=pai["PAI"],
                        color=pai["modality"],
                        hover_data=pai.columns)
        fig.write_html(self.pai_scatterplot_path)


    def save_random_sentences(self,):
        # shuffle dataset
        self.corpus_df = shuffle(self.corpus_df)
        self.corpus_df.reset_index(inplace=True, drop=True)

        vfunct = np.vectorize(self.add_to_random_sentences)
        vfunct(self.corpus_df.cw_df_path)

        logging.info(len(self.random_sentences))   
        logging.info(len(self.random_sentences['sight']))
        self.save_file(self.random_sentences, self.random_sentences_path)
        
        logging.info(len(self.random_contexts))
        self.save_file(self.random_contexts, self.random_contexts_path)

        logging.info(len(self.random_seed_words))
        self.save_file(self.random_seed_words, self.random_seed_words_path)


    def add_to_random_sentences(self, cw_df_path):
        start_time = time.time()
        cw_df = self.read_file(cw_df_path)
        descriptors = self.read_file(self.filtered_descriptors_path)
        for modality, modality_descriptors in descriptors.items():
            cw_df.apply(lambda row: self.find_random_sentences(row, modality_descriptors, modality), axis = 1)
        end_time = time.time()
        logging.info(end_time - start_time)

    def find_random_sentences(self, row, modality_descriptors, modality):
        if row['sense_name'] == modality:
            
            for word in row['context_window']:
                
                if word in modality_descriptors:

                    if word in self.random_sentences[modality] and len(self.random_sentences[modality][word]) < 3 and row['sentence'] not in self.random_sentences[modality][word]:
                        self.random_sentences[modality][word].append(row['sentence'])
                        self.random_seed_words[modality][word].append(row['seed_word'])
                        self.random_contexts[modality][word].append(row['context_window'])
                    else:
                        self.random_sentences[modality][word] = [row['sentence']]
                        self.random_seed_words[modality][word] = [row['seed_word']]
                        self.random_contexts[modality][word] = [row['context_window']]
                        
    def calculate_tf_idf(self, method):
        context_windows = self.read_file(self.context_windows_path)
        word_freq = self.read_file(self.word_freq_path) 
        self.word_freq = word_freq
        if method == 'method_one':
            tfs = {}
            for modality, windows in context_windows.items():
                words = [word for window in windows for word in window]
                total_words = len(words)
                counter = {}
                for word in words: 
                    if word in counter: 
                        counter[word] += 1
                    else:
                        counter[word] = 1
                counter = {key:value/total_words for key, value in counter.items()}
                tfs[modality] = counter
            
            tf_idf = {}
            for modality, tf in tfs.items():
                idf = {}
                for term, freq in tf.items():
                    count = 1
                    for sense_name, windows in context_windows.items():
                        words = [word for window in windows for word in window]
                        if term in words: 
                            count += 1
                    idf[term] = freq * np.log2(5/count)
                tf_idf[modality] = idf
            self.covert_tf_idf_dict_to_df(tf_idf)
            self.save_tf_idf_plot()
                
        if method == 'method_two':
            tfs = {}
            for modality, windows in context_windows.items():
                words = [word for window in windows for word in window]
                total_words = len(words)
                counter = {}
                for word in words: 
                    if word in counter: 
                        counter[word] += 1
                    else:
                        counter[word] = 1
                counter = {key:value/total_words for key, value in counter.items()}
                tfs[modality] = counter 
            
            if (not exists(self.nr_word_freq_path)):
                logging.info('Extracting Non-Relevant Windows')
                vfunct = np.vectorize(self.extract_non_relevant_windows)
                vfunct(self.corpus_df.tokenized_path, self.corpus_df.nr_cw_path) 
                
                logging.info('Counting Non-Relevant Words')
                vfunct = np.vectorize(self.count_non_relevant_words)
                vfunct(self.corpus_df.nr_cw_path)
                self.save_file(self.nr_word_freq, self.nr_word_freq_path)
            
            self.nr_word_freq = self.read_file(self.nr_word_freq_path)
            total_nr_words = sum(self.nr_word_freq.values())
            self.nr_word_freq = {key:value/total_nr_words for key, value in self.nr_word_freq.items()}
            tfs['nr_words'] = self.nr_word_freq
            
            tf_idf = {}
            for modality, tf in tfs.items():
                idf = {}
                for term, freq in tf.items():
                    count = 1
                    for sense_name, windows in context_windows.items():
                        words = [word for window in windows for word in window]
                        if term in words: 
                            count += 1
                    if term in tfs['nr_words']:
                        count += 1
                    idf[term] = freq * np.log2(6/count)
                tf_idf[modality] = idf
            self.covert_tf_idf_dict_to_df(tf_idf)
            self.save_tf_idf_plot()
                
        if method == 'method_three':
            corpus = []
            for modality, window_set in context_windows.items():
                document = [str(word[0]) + "<>" + str(word[1]) for window in window_set for word in window]
                corpus.append(document)
            tf_idf_vect = TfidfVectorizer(lowercase= False, analyzer=lambda x:[w for w in x if w.split('<>')[0] not in stopwords])
            X_train_tf_idf = tf_idf_vect.fit_transform(corpus)
            terms = tf_idf_vect.get_feature_names_out()
            tf_idf = self.dtm2dict(X_train_tf_idf ,terms)
            self.covert_tf_idf_dict_to_df(tf_idf)
            self.save_tf_idf_plot()
            
        
    def covert_tf_idf_dict_to_df(self, tf_idf):
        filtered_descriptors = self.read_file(self.filtered_descriptors_path)
        
        for modality, descriptor_dict in filtered_descriptors.items():
            for word, count in descriptor_dict.items():
                if word in tf_idf[modality]:
                    self.tf_idf = self.tf_idf.append({'modality':modality,
                                                      'word' : word,
                                                      'TF-IDF' : tf_idf[modality][word],
                                                      'word_freq': self.word_freq[word]}, ignore_index = True)
        logging.info(self.tf_idf.head())
        self.save_file(self.tf_idf, self.tf_idf_path)
        
        
    def save_tf_idf_plot(self,):

        fig = px.scatter(self.tf_idf, 
                    x=self.tf_idf["word_freq"],
                    y=self.tf_idf["TF-IDF"],
                    color=self.tf_idf["modality"],
                    hover_data=self.tf_idf.columns)
        fig.write_html(self.tf_idf_scatter_plot)
         
            
    def extract_non_relevant_windows(self, tokenized_path, nr_cw_path):
        start_time = time.time()
        window_size = 4
        punct = ['.', ',', ':', ';', '"', '!', "?"]
        ok_pos = ['ADJ', 'ADV', 'NOUN', 'VERB']
        start_time = time.time()
        text = self.read_file(tokenized_path)
        non_relevant_windows = []

        for i, sent in enumerate(text):
            has_seed_word = False 
            for j, word in enumerate(sent):
                if word in self.seed_words['seed_word'].tolist():
                    has_seed_word = True 
                    break
            if has_seed_word:
                non_relevant_windows.append(sent)
                        
        end_time = time.time()
        logging.info(end_time - start_time)
        self.save_file(non_relevant_windows, nr_cw_path)
        
    def count_non_relevant_words(self, nr_cw_path):
        start_time = time.time()
        nr_cws = self.read_file(nr_cw_path)
        words = [word for sent in nr_cws for word in sent]
        for w in words: 
            if w in self.nr_word_freq:
                self.nr_word_freq[w] += 1
            else:
                self.nr_word_freq[w] = 1
        end_time = time.time()
        logging.info(end_time - start_time)
        
        
    def dtm2dict(self, wm, feat_names):
        wm = wm.toarray()
        tf_idf = {}
        for i, row in enumerate(wm):
            doc_name = self.modalities[i]
            idf = {}
            for j, value in enumerate(row): 
                word = (feat_names[j].split('<>')[0], feat_names[j].split('<>')[1])
                idf[word] = value
            tf_idf[doc_name] = idf 
        return tf_idf 

                
def main(args):
    corpus = Corpus(args)
    if args.tokenize == 'True':
        logging.info('--Tokenizing--')
        corpus.tokenize_doc()
    if args.process == 'True':
        logging.info('--Processing Texts--')
        corpus.process_doc()
    if args.save_context_windows == 'True':
        logging.info('--Combining and Saving Context Windows--')
        corpus.combine_context_windows()
    if args.save_descriptors == 'True':
        logging.info('--Combining and Saving Descriptors--')
        corpus.combine_desciptors()
    if args.apply_descriptor_threshold == 'True':
        logging.info('--Filtering Out Descriptors--')
        corpus.apply_descriptor_threshold(args.descriptor_threshold)
    if args.calculate_PAI == 'True':
        logging.info('--Calculating PAI--')
        corpus.calculate_PAI()
    if args.create_PAI_visuals == 'True':
        logging.info('--Creating PAI Plots--')
        corpus.create_PAI_visuals()
    if args.save_random_sentences == 'True':
        logging.info('--Saving random_sentences--')
        corpus.save_random_sentences() 
    if args.tf_idf_method != 'False':
        logging.info('--Calculate TF-TDF--')
        corpus.calculate_tf_idf(args.tf_idf_method)

    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_type', type=str, default='None', help='century of author birth')
    parser.add_argument('--tokenize', type=str, default='False', help='tokenize corpus')
    parser.add_argument('--process', type=str, default='False', help='process corpus')
    parser.add_argument('--save_context_windows', type=str, default='False', help='save context windows')
    parser.add_argument('--save_descriptors', type=str, default='False', help='save descriptors')
    parser.add_argument('--apply_descriptor_threshold', type=str, default='False', help='apply descriptor threshold')
    parser.add_argument('--descriptor_threshold', type=int, default=30, help='threshold value')
    parser.add_argument('--calculate_PAI', type=str, default='False', help='calculate PAI value')
    parser.add_argument('--create_PAI_visuals', type=str, default='False', help='create PAI visuals')
    parser.add_argument('--save_random_sentences', type=str, default='False', help='save random sentences')
    parser.add_argument('--tf_idf_method', type=str, default='False', help='calculate tf idf')
    args = parser.parse_args()
    main(args)    