import sys, time, argparse, re, gensim, math
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import numpy as np
import faulthandler
import logging
from os.path import exists
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import matplotlib.pyplot as plt

logging.basicConfig(level = logging.DEBUG)

class Model: 
    def __init__(self, args):
        self.corpus_df = pd.read_csv('../data/english_fiction_metadata.csv')
        self.corpus_type = args.corpus_type
        if args.corpus_type == '1700':
            self.corpus_df = self.corpus_df.loc[(self.corpus_df['Author Birth Century'] == '1700')]
        elif args.corpus_type == '1800':
            self.corpus_df = self.corpus_df.loc[(self.corpus_df['Author Birth Century'] == '1800')]
        elif args.corpus_type == '1900':
            self.corpus_df = self.corpus_df.loc[(self.corpus_df['Author Birth Century'] == '1900')]
        self.modalities = ['sight', 'hear', 'touch', 'taste', 'smell']
        self.top_descriptors = [0, 200, 300, 500]
        self.filter_type = args.create_pca_graph 
        self.tf_idf_method = args.tf_idf_method
        
        self.context_windows_path = '../data/' + args.corpus_type + '/context_windows.pickle'
        self.descriptors_path = '../data/' + args.corpus_type + '/descriptors.pickle'
        self.filtered_descriptors_path = '../data/' + args.corpus_type + '/filtered_descriptors.pickle'
        self.word_freq_path = '../data/' + args.corpus_type + '/word_freq.pickle'
        self.pai_df_path = '../data/' + args.corpus_type + '/pai_df.pickle'
        self.model_path = '../data/' + args.corpus_type + '/word_embedding_model.pickle'
        self.principal_components_path = '../data/' + args.corpus_type + '/pca.pickle'
        self.pai_cutoff_path = '../data/' + args.corpus_type + '/pai_cutoff.csv'
        self.top_indices_path = '../data/' + args.corpus_type + '/top_indices.pickle'
        self.top_descriptors_path = '../data/' + args.corpus_type + '/top_descriptors.pickle'
        self.pca_path = '../data/' + args.corpus_type + '/pca_path.pickle'
        self.random_sentences_path = '../data/' + args.corpus_type + '/random_sentences.pickle'
        self.random_contexts_path = '../data/' + args.corpus_type + '/random_contexts.pickle'
        self.random_seed_words_path = '../data/' + args.corpus_type + '/random_seed_words.pickle'
        self.tf_idf_path = '../data/' + args.corpus_type + '/tf_idf_' + self.tf_idf_method + '.pickle'
        
        self.context_windows =  {'sight': [], 'hear': [], 'touch': [], 'taste': [], 'smell': []}
        self.descriptors = {'sight': {}, 'hear': {}, 'touch': {}, 'taste': {}, 'smell': {}}
        self.filtered_descriptors =  {'sight': {}, 'hear': {}, 'touch': {}, 'taste': {}, 'smell': {}}
        self.word_freq = {}
        self.pai_df = pd.DataFrame(columns = ['word', 'modality', 'total_freq', 'total_freq_in_sense', 'PAI'])
        self.principal_components = None
        self.random_sentences =  {'sight': {}, 'hear': {}, 'touch': {}, 'taste': {}, 'smell': {}}
        self.random_contexts = {'sight': {}, 'hear': {}, 'touch': {}, 'taste': {}, 'smell': {}}
        self.random_seed_words = {'sight': {}, 'hear': {}, 'touch': {}, 'taste': {}, 'smell': {}}
        self.tf_idf = pd.DataFrame(columns = ['word', 'modality', 'TF-IDF', 'word_freq'])

        
    def read_file(self, input_path):
        logging.info('Opened: ' + input_path)
        with open(input_path, "rb") as f:
            input_obj = pkl.load(f)
            return input_obj

    
    def save_file(self, output_obj, output_path):
        with open(output_path, 'wb') as f:
            pkl.dump(output_obj, f)
        logging.info('Saved to: ' + output_path)
        
    
    def train_model(self,):
        start_time = time.time()
        self.context_windows = self.read_file(self.context_windows_path)
        context_windows_by_sense = sum(self.context_windows.values(), [])
        model = gensim.models.Word2Vec(context_windows_by_sense, min_count=1, epochs=30, window=5, vector_size=200, workers=4, sg=0)
        model.save(self.model_path)
        end_time = time.time()
        logging.info(end_time - start_time)     
        
    def calculate_pca(self,):
        
        filtered_descriptors = self.read_file(self.filtered_descriptors_path)
        model = gensim.models.Word2Vec.load(self.model_path)
        k = 2 
        filtered_descriptors_df = pd.DataFrame(columns=['modality', 'descriptor', 'count'])
        for modality, descriptor_dict in filtered_descriptors.items():
            descriptor_list = [[modality, word, count] for word, count in descriptor_dict.items()]
            filtered_descriptors_df = filtered_descriptors_df.append(pd.DataFrame(descriptor_list, columns = ['modality', 'descriptor', 'count']))
            filtered_descriptors_df = filtered_descriptors_df.reset_index(drop=True)
            logging.info(filtered_descriptors_df.head())
            logging.info(filtered_descriptors_df.shape)
           
        logging.info('--Calculating Distance Matrix--')
        vectors = self.get_word_vectors(model, filtered_descriptors_df)

        logging.info('--Getting similarity matrix--')
        similarity_matrix = self.get_similarity_matrix(vectors)

        logging.info('--Running PCA--')
        self.principal_components = self.run_pca(similarity_matrix, k)
            
        self.save_file(self.principal_components, self.principal_components_path)
            
    def get_word_vectors(self, model, descriptor_df):
        vectors = []
        wv = model.wv
        not_found = 0
        for idx, row in descriptor_df.iterrows():
            try:
                vec_index = wv.key_to_index[row['descriptor']]
                vectors.append(wv[vec_index])
            except KeyError:
                logging.info(key, 'not found')
                not_found += 1
                continue

        n = len(vectors)
        logging.info("n: " + str(n))
        logging.info("not_found: " + str(not_found))

        return vectors       
    
    def get_similarity_matrix(self, vectors):
        n = len(vectors)
        distances = np.zeros((len(vectors[:n]), len(vectors[:n])))

        for idx1, vec1 in enumerate(tqdm(vectors[:n])):
            for idx2, vec2 in enumerate(vectors[idx1:n]):
                if idx2 == 0:
                    distances[idx1][idx1] = 0
                    continue
                # calculate the pearson correlation 
                p = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                # calculate the distance between vectors 
                dist = abs(0.5 * (1 - p))

                distances[idx1][idx1 + idx2] = dist
                distances[idx1 + idx2][idx1] = dist

        return distances
    
    def run_pca(self, distances, k):
        dist_matrix = pd.DataFrame(distances)
        x = dist_matrix.loc[:, ].values
        x = StandardScaler().fit_transform(x)
        reduction_model = PCA(n_components=k)
        pca = reduction_model.fit_transform(x)
        total_var = reduction_model.explained_variance_ratio_.sum() * 100
        logging.info("Total variance explained:" + str(total_var))
        return pca
    
    def create_pca_graph(self):
        
        # order descriptors by filter_type
        descriptors_df = self.order_descriptors()
        
        for top_descriptors_num in self.top_descriptors:
            logging.info("Creating PCA Graph For " + str(top_descriptors_num) + " Descriptors.")
        
            # get top n descriptors
            top_indices, top_descriptors = self.get_top_descriptors(descriptors_df, top_descriptors_num)

            # create pca dataframe 
            pca = self.read_file(self.principal_components_path)
            principleDf = self.create_pca_dataframe(pca)
            top_xs, top_ys  = self.get_xy(principleDf, descriptors_df, top_indices, top_descriptors_num)

            # create visual
            self.create_visual(top_xs, top_descriptors, principleDf, top_indices, top_descriptors_num)
        
        
    def order_descriptors(self,):
        filtered_descriptors = self.read_file(self.filtered_descriptors_path)
        descriptors_df = pd.DataFrame()
        
        if self.filter_type == 'PAI':
            filter_df = self.read_file(self.pai_df_path)
        elif self.filter_type == 'TF-IDF': 
            filter_df = self.read_file(self.tf_idf_path)

        for modality in self.modalities:
            df = filter_df.loc[(filter_df['modality'] == modality)]
            df = df.sort_values(by=[self.filter_type], ascending=False)
            descriptors_df = pd.concat([descriptors_df, df])

        descriptors_df = descriptors_df.sort_values(self.filter_type, ascending=False)

        return descriptors_df
    
    def get_top_descriptors(self, descriptors_df, top_descriptors_num):
        n = top_descriptors_num
        if n == 0:
            n = descriptors_df.shape[0]
        n_each = math.ceil(n / len(self.modalities))
        top_each = []
        for modality in self.modalities:
            descriptors_sense = descriptors_df[descriptors_df["modality"] == modality]
            top_each.append(descriptors_sense.head(n_each).dropna()) 
        top_descriptors = pd.concat(top_each).sort_values(self.filter_type, ascending=False)
        top_indices = top_descriptors.index.values.tolist()
        return top_indices, top_descriptors
    
    def create_pca_dataframe(self, principal_components):
        k = 2
        principleDf = pd.DataFrame()
        columns = ["Principal Component " + str(x) for x in range (1, k + 1)]
        item = pd.DataFrame(data=principal_components, columns=columns)
        principleDf = pd.concat([principleDf, item])
        principleDf = principleDf.reset_index(drop=True)
        return principleDf
        
    def get_xy(self, principleDf, descriptors_df, top_indices, top_descriptors_num):
        num_words_display = 20
        k = 2
        top_xs_path = '../data/' + args.corpus_type + '/' + args.corpus_type + '_top_xs_' + str(top_descriptors_num) + '_' + self.filter_type  + '_' + self.tf_idf_method + '.pickle'
        top_ys_path = '../data/' + args.corpus_type + '/' + args.corpus_type + '_top_ys_' + str(top_descriptors_num) + '_' + self.filter_type  + '_' + self.tf_idf_method + '.pickle'
        descrip_components = descriptors_df.merge(principleDf, how='outer', left_index=True, right_index=True)
        
        all_components = [[] for i in range(k)]
        for i in range(k):
            col_name = 'Principal Component ' + str(i + 1)
            component = pd.DataFrame(pd.concat([descrip_components.loc[:, col_name].nsmallest(num_words_display),
                                                descrip_components.loc[:, col_name].nlargest(
                                                    num_words_display)])).drop_duplicates()
            component = component.sort_values(col_name)
            indices = component.index.values.tolist()
            temp = []
            for idx in indices:
                pca_value = str(round(component.loc[idx, col_name], 3))
                w = descrip_components.loc[idx, :]['word'][0]
                s = descrip_components.loc[idx, :]["modality"]
                try:
                    temp.append(w + " (" + s + "):   " + pca_value)
                except TypeError:
                    logging.info(w, s)
                    logging.info(pca_value)
                    continue
            all_components[i] += temp

        top_xs = principleDf.iloc[top_indices, :].loc[:, 'Principal Component 1']
        top_ys = principleDf.iloc[top_indices, :].loc[:, 'Principal Component 2']
        self.save_file(top_xs, top_xs_path)
        self.save_file(top_ys, top_ys_path)
        return top_xs, top_ys
    
    def create_visual(self, top_xs, top_descriptors, principleDf, top_indices, top_descriptors_num):
        num_contexts = 3
        all_df = []
        k = 2
        
        pca_visual_path = '../visuals/' + args.corpus_type + '/' + args.corpus_type + '_pca_plot_' + str(top_descriptors_num) + '_' + self.filter_type + '_' + self.tf_idf_method + '.html'
        pca_visual_pdf = '../visuals/' + args.corpus_type + '/' + args.corpus_type + '_pca_plot_' + str(top_descriptors_num) + '_' + self.filter_type  + '_' + self.tf_idf_method + '.pdf'
        all_top_descriptors_path = '../git_data/' + args.corpus_type + '/' + args.corpus_type + '_all_top_descriptors_' + str(top_descriptors_num) + '_' + self.filter_type  + '_' + self.tf_idf_method + '.pickle'
        
        top_descriptors = self.set_word_info(top_descriptors, top_xs )
            
        df = pd.concat([top_descriptors, principleDf.iloc[top_indices, :]], axis=1)
        all_df.append(df)
        
        all_top_descriptors = pd.concat(all_df)
        
        self.save_file(all_top_descriptors, all_top_descriptors_path)

        fig = px.scatter(
            all_top_descriptors,
            color=all_top_descriptors["modality"],
            color_discrete_map={'sight': '#1f77b4', 'hear': '#2ca02c', 'taste': '#d62728', 'smell': '#ff7f0e', 'touch': '#9467bd'},
            x=all_top_descriptors.columns[-2],
            y=all_top_descriptors.columns[-1],
            hover_data=all_top_descriptors.columns[:-1 * k],
            custom_data=all_top_descriptors.columns[:-1 * k],
            title=self.corpus_type + " Literary Period Top " +  + str(top_descriptors_num) + " Descriptors - 2 Component PCA"
        fig.update_traces(marker=dict(size=12,))
        logging.info('--Saving PCA Graph--')
        fig.write_html(pca_visual_path)

        fig, ax = plt.subplots()
        for i,d in all_top_descriptors.groupby('modality'):
            ax.scatter(d['Principal Component 1'], d['Principal Component 2'], label=i)
        plt.title(self.corpus_type + " Literary Period Top " +  + str(top_descriptors_num) + " Descriptors - 2 Component PCA")
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.savefig(pca_visual_pdf)
        
        logging.info("Done! Figure saved to " + pca_visual_path)
        
    def set_word_info(self, top_descriptors, top_xs):
        self.random_sentences = self.read_file(self.random_sentences_path)
        self.random_contexts = self.read_file(self.random_contexts_path)
        self.random_seed_words = self.read_file(self.random_seed_words_path)

        for idx in top_xs.index:
            start_time = time.time()
            label = top_descriptors.loc[idx, :]['word'][0]
            pos = top_descriptors.loc[idx, :]['word'][1]
            word = (label, pos)
            modality = top_descriptors.loc[idx, :]['modality']
            
            for i in range(0, 3):
                if i >= len(self.random_sentences[modality][word]):
                    top_descriptors.at[idx, "Random Sentence " + str(i + 1)] = "NA"
                else:
                    sent = self.random_sentences[modality][word][i]
                    sent = [word[0] for word in sent]
                    sent2 = []
                    for w in sent:
                        if len(sent2) % 10  == 0:
                            sent2.append('<br>')
                            sent2.append(w)
                        else:
                            sent2.append(w)
                    top_descriptors.at[idx, "Random Sentence " + str(i + 1)] =  ' '.join(sent2)
                if i >= len(self.random_contexts[modality][word]):
                    top_descriptors.at[idx, "Random Context Window " + str(i + 1)] = "NA"
                else:
                    context = self.random_contexts[modality][word][i]
                    context = [word[0] for word in context]
                    top_descriptors.at[idx, "Random Context Window " + str(i + 1)] = ' '.join(context)
                if i >= len(self.random_seed_words[modality][word]):
                    top_descriptors.at[idx, "Random Seed Words " + str(i + 1)] = "NA"
                else:
                    top_descriptors.at[idx, "Random Seed Words " + str(i + 1)] = str(self.random_seed_words[modality][word][i])

            end_time = time.time()
            logging.info(end_time - start_time)     
                
        return top_descriptors
        
        
def main(args):
    model = Model(args)
    if args.run_model == 'True':
        logging.info('--Training Model--')
        model.train_model()
    if args.calculate_pca == 'True':
        logging.info('--Running PCA--')
        model.calculate_pca()
    if args.create_pca_graph != 'False':
        logging.info('--Create pca graph--')
        model.create_pca_graph()

        
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_type', type=str, default='None', help='century of author birth')
    parser.add_argument('--run_model', type=str, default='False', help='run word embedding model')
    parser.add_argument('--calculate_pca', type=str, default='False', help='run pca')
    parser.add_argument('--create_pca_graph', type=str, default='False', help='create pca graph')
    parser.add_argument('--tf_idf_method', type=str, default='False', help='select tf_idf method to use')
    args = parser.parse_args()
    main(args)