
# coding: utf-8

import requests, json, sys, time, re
from bs4 import BeautifulSoup
from datetime import datetime
import sqlalchemy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
from pyspark.mllib.recommendation import ALS
from pyspark import SparkContext
from pathlib import Path
from pprint import pprint
from tqdm import tqdm


path_app_info = Path.cwd() / 'data' / 'appDetail.txt'
path_app_stats = Path.cwd() / 'data' / 'steamspy_appinfo.json'
path_user_inventory = Path.cwd() / 'data' / 'crawled_user_inventory.txt'

engine = sqlalchemy.create_engine('mysql+pymysql://root:591743372a//@127.0.0.1/game_recommendation?charset=utf8mb4')
engine.execute('ALTER DATABASE game_recommendation CHARACTER SET=utf8mb4')


dic_steamspy = {'owners':{}}
with open(path_app_stats, 'r') as f:
    for line in f.readlines():
        temp_dic = json.loads(line)
        dic_steamspy['owners'].update({temp_dic.get('appid'):temp_dic.get('owners')})


with open(path_app_info, 'r') as f:
    dic_steam_app = {'initial_price':{}, 'name':{}, 'score':{}, 'windows':{}, 'mac':{}, 'linux':{}, 'type':{}, 'release_date':{}, 'recommendation':{}, 'header_image':{}, 'currency':{}, 'success':{}}
    dic_about_the_game = {}
    lst_raw_string = f.readlines()
    for raw_string in tqdm(lst_raw_string):
        steam_id, response = next(iter(json.loads(raw_string).items()))
        app_data = response.get('data', {})
        if not app_data:
            dic_steam_app['success'].update({steam_id:False})
        else:
            initial_price = app_data.get('price_overview', {}).get('initial')
            currency = app_data.get('price_overview', {}).get('currency')
            if app_data.get('is_free') == True:
                initial_price = 0
            app_name = app_data.get('name')
            critic_score = app_data.get('metacritic', {}).get('score')
            app_type = app_data.get('type')
            for (platform, is_supported) in app_data.get('platforms').items():
                if is_supported:
                    dic_steam_app[platform].update({steam_id:1})
            if not app_data.get('release_date',{}).get('coming_soon'):
                about_the_game = app_data.get('about_the_game')
                soup = BeautifulSoup(about_the_game, 'lxml')
                game_description = re.sub(r'(\s+)', ' ', soup.text).strip()  # This part is to extract out the pure text information from the html formatted original string with all the tags.
                dic_about_the_game.update({steam_id:game_description})
                release_date = app_data.get('release_date', {}).get('date')
                if release_date:
                    if re.search(',', release_date) == None:
                        release_date = datetime.strptime(release_date, '%b %Y')
                    else:
                        try:
                            release_date = datetime.strptime(release_date, '%b %d, %Y')
                        except:
                            release_date = datetime.strptime(release_date, '%d %b, %Y')
            recommendation = app_data.get('recommendations',{}).get('total')
            header_image = app_data.get('header_image')
            dic_steam_app['initial_price'].update({steam_id:initial_price})
            dic_steam_app['currency'].update({steam_id:currency})
            dic_steam_app['name'].update({steam_id:app_name})
            dic_steam_app['score'].update({steam_id:critic_score})
            dic_steam_app['type'].update({steam_id:app_type})
            dic_steam_app['release_date'].update({steam_id:release_date})
            dic_steam_app['recommendation'].update({steam_id:recommendation})
            dic_steam_app['header_image'].update({steam_id:header_image})
            
df_steam_app = pd.DataFrame(dic_steam_app)
df_steam_app.initial_price = df_steam_app.initial_price / 100.0
df_steam_app.index.name = 'steam_appid' # Due to how this df is created from the big dict of dict, the app_id wil become the index and sql won't accept indexed df, so need to reset_index after give the index a name.
df_steam_app['windows'] = df_steam_app.windows.fillna(0)
df_steam_app['mac'] = df_steam_app.mac.fillna(0) 
df_steam_app['linux'] = df_steam_app.linux.fillna(0)
df_steam_app = df_steam_app[['name', 'type', 'currency', 'initial_price', 'release_date', 'score', 'recommendation', 'windows', 'mac', 'linux', 'success','header_image']]
df_steam_app.reset_index(inplace=True)
df_steam_app.success.fillna(True, inplace=True)
df_steam_app.to_sql('tbl_app_info', engine, if_exists='replace', index=False)

# This is for creating the game description dict.
with open(path_app_info, 'r') as f:
    dic_about_the_game = {}
    lst_raw_string = f.readlines()
    for raw_string in tqdm(lst_raw_string):
        steam_id, response = next(iter(json.loads(raw_string).items()))
        app_data = response.get('data', {})
        if app_data:
            about_the_game = app_data.get('about_the_game')
            soup = BeautifulSoup(about_the_game, 'lxml')
            game_description = re.sub(r'(\s+)', ' ', soup.text).strip()  # This part is to extract out the pure text information from the html formatted original string with all the tags.
            dic_about_the_game.update({steam_id:game_description})

path_user_inventory = Path.cwd() / 'data' / 'crawled_user_inventory.txt'
dic_user_favorite_app = {}
with open(path_user_inventory, 'r') as f:
    for raw_string in f.readlines():
        user_id, lst_inventory = next(iter(json.loads(raw_string).items()))
        if lst_inventory:
            most_played_app_id = sorted(lst_inventory, key=lambda k: k['playtime_forever'])[-1].get('appid')
        else:
            most_played_app_id = None
        dic_user_favorite_app.update({user_id:most_played_app_id})
df_user_favorite_app = pd.Series(dic_user_favorite_app).reset_index()
df_user_favorite_app.columns = ['steam_user_id', 'favorite_app']
df_user_favorite_app.to_sql('tbl_user_favorite_app', engine, if_exists='replace', index=False)

# build recommendation models
df_steam_app = pd.read_sql('tbl_app_info', engine)
# It's kinda funky when using datetime stuff with {}.format in the query expression, need to use "{}" instead of just {}
df_valid_games = df_steam_app.query('success==True and type=="game" and release_date<=datetime.today().date() and initial_price >=0')
set_valid_game_id = set(df_valid_games.steam_appid)

df_popularity_based_results = pd.Series(dic_steamspy.get('owners'))
df_popularity_based_results.name = 'owners'
df_popularity_based_results.index.name = 'steam_appid'
df_popularity_based_results = df_popularity_based_results.reset_index()
df_popularity_based_results = df_popularity_based_results.sort_values('owners',ascending=False)
df_popularity_based_results.to_sql('tbl_results_popularity_based', engine, if_exists='replace')


#content based recommendation | done. Summary: TfidfVectorizer to summarize every document, then use linear_kernel(cosine similarity kernel) to compute pair-wise similarity of the documents and for each document pick out the most similar ones.
print('content based recommendation')
for i in set(dic_about_the_game.keys())-set(df_valid_games.steam_appid):  # iter over set will work fast with /random order
    del dic_about_the_game[i]
    
tfidf = TfidfVectorizer(strip_accents = 'unicode', stop_words='english').fit_transform(dic_about_the_game.values())
lst_app_id = dic_about_the_game.keys()
dic_recommend = {}
for index in tqdm(range(tfidf.shape[0])):  #index is the common chain of dict_keys and dict_values. So loop over this index.
    cosine_similarities = linear_kernel(tfidf[index:index+1], tfidf).flatten() # TfidfVectorizer's default norm parameter is set to 'l2', so the tfidf's row vectors are already l2 normalized, feeding l2 normalized vectors to linear_kernal is exactly the same as feeding non-normalized vectors to cosine_similarity.
    related_docs_indices = cosine_similarities.argsort()[-2:-22:-1] # the largest cosine_similarity value would be the vector itself, so start with the second largest: -2. And we want 20 highly similar indices, considering that step is -1, the end anchor would be first anchor minus number of queried elements count: -2-20=-22. If not negative step, it would be plus queried elements count.
    dic_recommend.update({list(lst_app_id)[index]:[list(lst_app_id)[i] for i in related_docs_indices]})

df_content_based_results = pd.DataFrame(dic_recommend).T # colunms --> index and index --> column by .T, this will position to to-recommend-to users as the row index and columns just sequence of number 0 to 19 denoting the level of recommendation.
df_content_based_results.index.name = 'steam_appid'
df_content_based_results = df_content_based_results.reset_index()
df_content_based_results.to_sql('tbl_results_content_based', engine, if_exists='replace')

# item based
print('item based recommendation')
dic_purchase = {}
with open(path_user_inventory, 'r') as f:
    lst_all = f.readlines()
    for i in tqdm(lst_all):
        user_id, user_inventory = next(iter(json.loads(i).items()))
        if user_inventory:
            dic_purchase[user_id] = {}
            for playtime_info in user_inventory:
                appid = playtime_info.get('appid')
                if str(appid) in set_valid_game_id:
                    dic_purchase[user_id].update({appid:1})
df_purchase = pd.DataFrame(dic_purchase).fillna(0)
purchase_matrix = df_purchase.values
lst_user_id = df_purchase.columns
lst_app_id = df_purchase.index

dic_recommend_item_based = {}
for index in tqdm(range(purchase_matrix.shape[0])): # This non-vectorized computer is bad practice. Actually bulk compute the kernel matrix and enumerate through it to get individual 1darray would be much faster.
    cosine_similarities = linear_kernel(purchase_matrix[index:index+1],purchase_matrix).flatten()
    lst_related_app = np.argsort(-cosine_similarities)[1:101] # argsort will put small first, we want large first, so sort negative value instead
    dic_recommend_item_based.update({lst_app_id[index]:[lst_app_id[i] for i in lst_related_app]})

df_item_based_result = pd.DataFrame(dic_recommend_item_based).T
df_item_based_result.index.name = 'steam_appid'
df_item_based_result = df_item_based_result.reset_index()
df_item_based_result.to_sql('tbl_results_item_based',engine, if_exists='replace')


#ALS model. This part needs to run in a PySpark notebook
print('ALS Model')
sc = SparkContext()

def parse_raw_string(raw_string):
    user_inventory = json.loads(raw_string)
    return user_inventory.items()[0]

def id_index(x):
    ((user_id,lst_inventory),index) = x
    return (index, user_id)

def create_tuple(x):
    ((user_id,lst_inventory),index) = x
    if lst_inventory != None:
        return (index, [(i.get('appid'), 1) for i in lst_inventory if str(i.get('appid')) in set_valid_game_id])
    else:
        return (index, [])

user_inventory_absolute_path = '/Users/jiaxu/Desktop/Game_Reco/Data/crawled_user_inventory.txt'

user_inventory_rdd = sc.textFile(user_inventory_absolute_path).map(parse_raw_string).zipWithIndex()
dic_id_index = user_inventory_rdd.map(id_index).collectAsMap()
# training_rdd = user_inventory_rdd.map(create_tuple).flatMapValues(lambda x: x).map(lambda (index,(appid,time)):(index,appid,time))
training_rdd = user_inventory_rdd.map(create_tuple).flatMapValues(lambda x: x).map(lambda index,appid,time:(index,appid,1))
model = ALS.train(training_rdd, 5)

dic_recommended = {}
for index in dic_id_index.keys():
    try:
        lst_recommended = [i.product for i in model.recommendProducts(index,10)]
        user_id = dic_id_index.get(index)
        dic_recommended.update({user_id:lst_recommended})
    except:
        pass


df_als_result = pd.DataFrame(dic_recommended).T
df_als_result.index.name = 'steam_user_id'
df_als_result.reset_index(inplace=True)
df_als_result.to_sql('tbl_results_als_based',engine,if_exists='replace',index=False)
