from nltk.tokenize import word_tokenize
print("Importing Gensim module")
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
print("done")
from nltk.corpus import stopwords
import string
import numpy as np



def build_doc2vec(df, df_reviews, train_indices, test_indices, vsize = 5):
	df_train = df.loc[train_indices, :]
	df_test = df.loc[test_indices, :]
	df_train.index = range(len(df_train))
	df_test.index = range(len(df_test))
	
	stop_words = set(stopwords.words('english')) 
	
	dt = df_train.copy(deep=True)
	
	translator = str.maketrans('', '', string.punctuation)
	
	texts = []
	for i in range(dt.shape[0]):
	  #print(i)
	  #All dfs where this app 
	  #print(dt.loc[i, 'App'])
	  nd = df_reviews[df_reviews['App'] == dt.loc[i, 'App']]
	  for review in nd['Translated_Review']:
	    #print(type(review))
	    sent = []
	    for word in word_tokenize(review.translate(translator).lower()):
	      if word not in stop_words:
	        sent.append(word)
	    
	    sent = " ".join(sent)
	    texts.append(sent)
	    
	documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
	
	model = Doc2Vec(documents, vector_size=vsize, window=3, min_count=1, workers=4)
	
	return model
	

def transform_df(doc2vec_model, df_r, df_reviews, vsize):
	df_common = df_r.copy(deep = True)
	translator = str.maketrans('', '', string.punctuation)
	stop_words = set(stopwords.words('english')) 

	for i in range(vsize):
	  df_common["v_" + str(i)] = [0] * df_common.shape[0]

	print("Transforming dataset")
	for i in range(df_common.shape[0]):
  
	  r_indices = np.where(df_reviews['App'] == df_common.loc[i, 'App'])
	  vectors = []
	  #r_indices into review dataset
	  #take review, model.infer, append to vectors
	  #print(common_dataset.loc[i, 'App'], r_indices[0])
	  for j in r_indices[0]:
	    rv = df_reviews.loc[j, 'Translated_Review']
	    sent = []
	    for word in word_tokenize(rv.translate(translator).lower()):
	      if word not in stop_words:
	        sent.append(word)
	    
	    vectors.append(doc2vec_model.infer_vector(sent))
	  
	  
	  for k in range(vsize):
	    vdim = 0
	    for p in range(len(vectors)):
	      vdim += vectors[p][k]
	    
	    if len(vectors) > 0:
	      vdim = vdim / len(vectors)
	    #print(i, k, vdim)
	    df_common.loc[i, "v_" + str(k)] = vdim
	
	print("Done")
		
	return df_common