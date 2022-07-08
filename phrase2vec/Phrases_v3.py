import time, logging, sqlite3, sys
import pandas as pd 
import numpy as np 
import pickle as pkl

logging.basicConfig(level = logging.DEBUG)
conn = sqlite3.connect(r'/home/ashokan2/literary_periods/phrase2vec/context_window.db')
cursor = conn.cursor()
pragma = """PRAGMA case_sensitive_like=ON;"""
cursor.execute(pragma)

def convert(string):
    return string.replace("), (", "***").split("***")

def convert2(lst):
    return [tuple(map(replaceQuotes, itm.strip("()").split(","))) for itm in lst]

def replaceQuotes(string):
    return string.replace("'", "").strip()


class Corpus: 
    
    def __init__(self):
        self.corpus_df = pd.read_csv('../data/english_fiction_metadata.csv')
        self.context_windows_dict = '../data/full_corpus/context_windows_dict.pickle'
        self.context_windows_dict = {'sense_name': [], 'seed_word': [] , 'context_window': [], 'sentence': []}
        
    
    def read_file(self, input_path):
            logging.info('Opened: ' + input_path)
            with open(input_path, "rb") as f:
                input_obj = pkl.load(f)
                return input_obj     
            
    def save_file(self, output_obj, output_path):
        with open(output_path, 'wb') as f:
            pkl.dump(output_obj, f)
        logging.info('Saved to: ' + output_path)
            
    def save_context_windows(self,):
        vfunct = np.vectorize(self.insert_context_window_into_db)
        vfunct(self.corpus_df.cw_df_path)   
        
    def insert_context_window_into_db(self, cw_df_path):
        start_time = time.time()
        cw_df = self.read_file(cw_df_path)
        for idx, row in cw_df.iterrows():
            if row['context_window']:
                insert = """INSERT INTO CONTEXT_WINDOWS
                            (SENSE_NAME, SEED_WORD, CONTEXT_WINDOW, SENTENCE) 
                            VALUES (?, ?, ?, ?);"""
                data_tuple = (row['sense_name'], str(row['seed_word']), str(row['context_window']), str(row['sentence']))
                cursor.execute(insert, data_tuple)
                conn.commit()
        end_time = time.time()
        logging.info(end_time - start_time)
        
    def extract_noun_adjective_phrases(self,):
        sql_query = pd.read_sql_query("""SELECT * from CONTEXT_WINDOWS
                                 WHERE CONTEXT_WINDOW LIKE '%NOUN%'
                                 AND CONTEXT_WINDOW LIKE '%ADJ%'""", conn)
        df = pd.DataFrame(sql_query, columns = ['SENSE_NAME', 'SEED_WORD', 'CONTEXT_WINDOW', 'SENTENCE'])
        conn.close()
        df['test'] = df['CONTEXT_WINDOW'].str.strip('][')
        df['test'] = df['test'].apply(convert)
        df['test'] = df['test'].apply(convert2)
        df = df[df['test'].map(len) == 2]
        self.save_file(df['test'].tolist(), "../data/full_corpus/phrase_noun_adj_context_window_size_4.pickle")


corpus = Corpus()
# corpus.save_context_windows()

# cursor.execute('SELECT COUNT(*) from CONTEXT_WINDOWS')
# cur_result = cursor.fetchone()
# logging.info("There are " + str(cur_result) + " rows in the DB.")

# corpus.extract_noun_adjective_phrases()
conn.close()