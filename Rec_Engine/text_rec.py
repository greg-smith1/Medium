
import pandas as pd
from flask import current_app
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel



class RecommendationSystem(object):
    """Recommender that creates cosine similarity between objects based text.
    Takes in a data source that must be in csv format with 2 columns:
        an index ('item_id') and a column of text reviews ('text').
    Writes a csv containing the top 8 most similar items from the list.
    """
    def __init__(self, data_source):
        self.data = pd.read_csv(data_source)

    def populate(self):
        df = self.data
        model = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), 
                             min_df=0, stop_words='english')

        trans_matrix = model.fit_transform(df['text'])
        similarities = linear_kernel(tf_matrix, tf_matrix)
        return similarities

    def generate_similarity(self, matrix):
        df = self.data
        with open('similar_text_matrix.csv', 'w+') as similarity_matrix:
            similarity_matrix.write('self_id,id1,id2,id3,id4,id5,id6,id7,id8\n')
            for index, row in df.iterrows():
                similar_indices = matrix[index].argsort()[:-10:-1]
                similar_items = [(matrix[index][i], df['item_id'][i]) for i in similar_indices]
                new_items = sum(similar_items[:], ())
                # Unpack the new_items into the item names we're writing to csv
                # In this basic version we're not keeping the actual values
                self_score, self_id, score1, id1, score2, id2, score3, id3, score4, \
                id4, score5, id5, score6, id6, score7, id7, score8, id8 = new_items
                #item_name = df[(df['item_id'] == self_id)][:1]['item_name'].item()
                new_line = "{0},{1},{2},{3},{4},{5},{6},{7},{8}".format(
                    self_id,id1,id2,id3,id4,id5,id6,id7,id8)

                similarity_matrix.write('{new_line}\n'.format(new_line=new_line))

        
        return new_items, new_line