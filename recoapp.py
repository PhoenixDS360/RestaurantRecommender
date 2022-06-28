

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import codecs
codecs.register_error("strict", codecs.ignore_errors)
import matplotlib.pyplot as plt

import seaborn as sns
import os
import datetime

import warnings
warnings.filterwarnings('ignore')

import reverse_geocode
import time
import itertools
from datetime import datetime
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import OneHotEncoder
from sklearn.preprocessing   import OrdinalEncoder
from sklearn.preprocessing   import MinMaxScaler
from sklearn import metrics

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, f1_score

from scipy.stats import mode

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import streamlit as st


def calculate_duration(timings):
  '''
  Calculates the duration the restaurant functions in a day
  '''

  try:
    try:
      close_time = datetime.strptime(timings[1],"%I:%M%p")
    except ValueError:
      close_time = datetime.strptime(timings[1],"%I%p")
    try:
      open_time = datetime.strptime(timings[0],"%I:%M%p")
    except ValueError:
      open_time = datetime.strptime(timings[0],"%I%p")

    duration = close_time-open_time
    return round(abs(duration.total_seconds())/3600,2)

  except:
    return np.nan

def clean_food_items(food_tag):
  '''
  There are multiple food items supplied by a single vendor and some of these food items are multi-worded.
  This is a preprocessing step required before using tokenizer. 
  This method goes through the list of food items for every vendor and joins multi-worded items using underscore
  '''
  multiword_fixed = ['_'.join(food.split(' ')) for food in food_tag.split(',')]
  foodstring = ' '.join([str(food) for food in multiword_fixed])
  return foodstring


def process_vendors(location,filename):
  '''
  This method receives the name and location of the file containing the details of 100 vendors

  Creates new features for the vendors 
  -------------------------------------
  tag_counts - Number of food specialities available in the restaurant
  open_duration - Estimates the duration the restaurant is opened using the opening and closing time
  time features - Extracts the month and year the vendor is created/updated
  vendor_city, vendor_country - Using the location coordinates creates the city and country the vendor is located in

  Cleans feature data
  --------------------
  vendor_tag_name_cleaned - Multi-worded food items in vendor_tag_name joined using underscore 
  primary_tags_mod - Removes the constant prefix in values of primary_tags
  commission, language, vendor_locations - Fixes null values 
  '''
  vendors = pd.read_csv(location + filename, sep=',', encoding="utf-8",encoding_errors="ignore")
  vendors["id"] = vendors["id"].astype('str')
  vendors.rename(columns={'id':'vendor_id'},inplace=True)

  #Create count feature for vendor_tag attribute (Number of food specialities available in the restaurant)
  vendors["tag_counts"]=vendors["vendor_tag"].str.split(',').str.len()
  vendors["tag_counts"].fillna(0,inplace=True)
  
  #Preprocess Food specialities of the vendor
  vendors["vendor_tag_name"].fillna("null",inplace=True)
  vendors["vendor_tag_name_cleaned"]  = vendors["vendor_tag_name"].apply(clean_food_items)
  
  #Extract numerical part of the primary_tags
  vendors["primary_tags"].fillna('{"primary_tags":"0"}',inplace=True) #0 is assigned for null values
  vendors["primary_tags_mod"] = vendors["primary_tags"].str.split('{"primary_tags":"').str[1].str.split('"}').str[0]  

  vendors["OpeningTime"]= vendors["OpeningTime"].str.replace(" ","").str.replace(".",":").str.replace("::",":").str.replace("111","11").str.replace('08:00AM-11:45-','08:00AM-11:45PM')
  vendors["timings"] = vendors["OpeningTime"].str.split('-')
  vendors["open_duration"] = vendors["timings"].apply(calculate_duration)
  vendors["open_duration"].fillna(round(vendors["open_duration"].mean(),2),inplace=True)

  #Time Features
  vendors["year_vendor_created"] = pd.to_datetime(vendors["created_at"]).dt.year
  vendors["month_vendor_created"] = pd.to_datetime(vendors["created_at"]).dt.month
  vendors["year_vendor_updated"] = pd.to_datetime(vendors["updated_at"]).dt.year
  vendors["month_vendor_updated"] = pd.to_datetime(vendors["updated_at"]).dt.month

  vendors["month_vendor_created"] = vendors["month_vendor_created"].astype('str')
  vendors["month_vendor_updated"] = vendors["month_vendor_updated"].astype('str')
  vendors["year_vendor_updated"] = vendors["year_vendor_updated"].astype('str')
  vendors["year_vendor_created"] = vendors["year_vendor_created"].astype('str')

  #Fix nan values for commission, language
  vendors["commission"].fillna(1.0,inplace=True)
  vendors["language"].fillna("NA",inplace=True)
  vendors["language"] = vendors["language"].map({"NA":0,"EN":1})

  #Verify vendor locations
  #All the vendor locations are in Africa except for 2 which has incorrect coordinate values
  vendors_valid = vendors[(((vendors["latitude"] >= -90.0) & (vendors["latitude"] <= 90.0)) & ((vendors["longitude"] >= -180.0) & (vendors["longitude"] <= 180.0)))]

  coordinates = list(zip(vendors_valid["latitude"],vendors_valid["longitude"]))
  vendors_valid[["country_code","vendor_city","vendor_country"]] = pd.DataFrame(reverse_geocode.search(coordinates))
  vendors_valid["vendor_city"].fillna("null",inplace=True)
  vendors_valid["vendor_country"].fillna("null",inplace=True)

  #Extract the vendor info with incorrect coordinates and set their city and country values
  vendors_invalid = vendors[~(((vendors["latitude"] >= -90.0) & (vendors["latitude"] <= 90.0)) & ((vendors["longitude"] >= -180.0) & (vendors["longitude"] <= 180.0)))]
  vendors_invalid["vendor_city"]="invalid"
  vendors_invalid["vendor_country"]="invalid"
  vendors_valid = vendors_valid.append(vendors_invalid,ignore_index=True)

  #Remove the unwanted features
  vendor_dropcols = ["vendor_tag_name","vendor_tag","vendor_category_en","authentication_id","created_at","updated_at","OpeningTime","OpeningTime2","timings","primary_tags","is_akeed_delivering","country_code","open_close_flags","one_click_vendor","country_id","city_id","display_orders",'sunday_from_time1', 'sunday_to_time1', 'sunday_from_time2', 'sunday_to_time2', 'monday_from_time1', 'monday_to_time1', 'monday_from_time2', 'monday_to_time2', 'tuesday_from_time1', 'tuesday_to_time1', 'tuesday_from_time2', 'tuesday_to_time2', 'wednesday_from_time1', 'wednesday_to_time1', 'wednesday_from_time2', 'wednesday_to_time2', 'thursday_from_time1', 'thursday_to_time1', 'thursday_from_time2', 'thursday_to_time2', 'friday_from_time1', 'friday_to_time1', 'friday_from_time2', 'friday_to_time2', 'saturday_from_time1', 'saturday_to_time1', 'saturday_from_time2', 'saturday_to_time2']
  vendors_valid = vendors_valid.drop(columns=vendor_dropcols)
  return vendors_valid

@st.cache
def process_vendor_summary(location,filename):
  '''
  This method uses orders data given for the train customers, to extract features indicating the vendor performances on various parameters

  Min, Max and Average order_turnaround time for every vendor was calculated using features created_at and delivered_time

  Average order_preparation time for every vendor using their respective orders’ preparation_time feature.

  Other summary features:
  ------------------------
  count of customers across all orders
  count of unique customers
  number of payment options
  number of promo codes
  total discount amount

  '''
  orders = pd.read_csv(location + filename)
  orders.drop_duplicates(subset=['akeed_order_id'],inplace=True)
  orders['created_at'] = pd.to_datetime(orders["created_at"])
  orders['delivered_time'] = pd.to_datetime(orders['delivered_time']) 
  orders[['CID','LOC_NUM','VENDOR']] = orders['CID X LOC_NUM X VENDOR'].str.split(' X ',expand=True)

  orders["preparationtime"].fillna(orders["preparationtime"].groupby(orders["VENDOR"]).transform('mean'),inplace=True)
  orders["preparationtime"].fillna(orders["preparationtime"].mean(),inplace=True)
  orders["promo_code_discount_percentage"].fillna(0,inplace=True)
  orders["is_favorite"].fillna("null",inplace=True)

  #For few records the timestamps looks to be swapped...hence absolute function
  orders["order_turnaround"] = np.abs((orders["delivered_time"] - orders["created_at"]).apply(lambda x: x.total_seconds())) #total_seconds method belongs to timedelta object

  vendor_summary = orders.groupby(orders["VENDOR"]).agg({
    'customer_id':['count','nunique'],
    'payment_mode':['nunique'],
    'promo_code':['count'],
    'vendor_discount_amount':['sum'],
    'promo_code_discount_percentage':['mean'],
    'item_count':['median'],
    'grand_total':['median'],
    'driver_rating' : ['median'],
    'deliverydistance' : ['mean'],
    'preparationtime' :['mean'],
    'order_turnaround':['min','max','mean']}).reset_index()

  vendor_summary.columns = ['_'.join(col).strip() for col in vendor_summary.columns.values]
  vendor_summary.rename(columns={'VENDOR_':'vendor_id'},inplace=True)
  
  return vendor_summary

@st.cache
def process_customer_demo(location,filename):
  '''
  This method reads the customer demographics file and does basic cleanup of the features.

  '''

  customers = pd.read_csv(location + filename)
  customers["updated_at"] = pd.to_datetime(customers["updated_at"])
  customers["created_at"] = pd.to_datetime(customers["created_at"])

  #Remove duplicate records by extracting the last updated records
  customers_dedup = customers[customers["updated_at"] == customers.groupby(["akeed_customer_id"])['updated_at'].transform('max')]
  
  #remove trailing spaces in gender and convert to lower case
  customers_dedup.loc[:,"gender"] = customers_dedup["gender"].astype("str").str.rstrip().str.lower()

  #fix missing and incorrect values as 'unknown'
  customers_dedup.loc[~customers_dedup["gender"].isin(["male","female"]),"gender"] = "unknown"

  #time features
  customers_dedup['year_customer_created'] = customers_dedup['created_at'].dt.year
  customers_dedup['month_customer_created'] = customers_dedup['created_at'].dt.month
  customers_dedup['year_customer_updated'] = customers_dedup['updated_at'].dt.year
  customers_dedup['month_customer_updated'] = customers_dedup['updated_at'].dt.month
                                                    
  customers_dedup = customers_dedup[['akeed_customer_id', 'gender', 'verified', 'language', 'year_customer_created', 'month_customer_created', 'year_customer_updated', 'month_customer_updated']]

  return customers_dedup

@st.cache
def process_customer_location(location,filename):
  '''
  This method reads the customer locations file and does 
  
  Basic cleanup of the features

  Validates the latitude and longitude coordinates

  Using reverse_geocode package, it retrieves the customer location details using the coordinates

  '''
  
  locations = pd.read_csv(location + filename)
  locations["location_type"].fillna('Null',inplace=True)
  locations['location_type'] = locations['location_type'].map({'Null':0,'Home':1,'Work':2,'Other':3})

  locations["latitude"].fillna(locations.groupby(["customer_id"])["latitude"].transform("mean"),inplace=True)
  locations["longitude"].fillna(locations.groupby(["customer_id"])["longitude"].transform("mean"),inplace=True)
  
  locations["latitude"].fillna(locations["latitude"].mean(),inplace=True)
  locations["longitude"].fillna(locations["longitude"].mean(),inplace=True)

  locations_valid = locations[(((locations["latitude"] >= -90.0) & (locations["latitude"] <= 90.0)) & ((locations["longitude"] >= -180.0) & (locations["longitude"] <= 180.0)))]

  #Retrieve the city and country of those coordinates 
  coordinates = list(zip(locations_valid["latitude"],locations_valid["longitude"]))
  locations_valid[["country_code","customer_city","customer_country"]] = pd.DataFrame(reverse_geocode.search(coordinates))
  locations_valid.drop(columns=["country_code"],inplace=True)

  locations_valid["customer_city"].fillna("null",inplace=True)  #null for coordinates without city or country details
  locations_valid["customer_country"].fillna("null",inplace=True)
  locations_valid = locations_valid.append(locations[~(((locations["latitude"] >= -90.0) & (locations["latitude"] <= 90.0)) & ((locations["longitude"] >= -180.0) & (locations["longitude"] <= 180.0))) | locations["latitude"].isna()])
  locations_valid["customer_city"].fillna("invalid",inplace=True)
  locations_valid["customer_country"].fillna("invalid",inplace=True)

  #additional cleanup for deep learning models
  locations_valid["customer_country"] = locations_valid["customer_country"].apply(lambda x: '_'.join([ele for ele in x.split(' ')]))
  locations_valid["customer_city"] = locations_valid["customer_city"].apply(lambda x: '_'.join([ele for ele in x.split(' ')]))
  return locations_valid

def haversine_np(longitude1, latitude1, longitude2, latitude2):
    """
    Calculate the circle distance between two points on the earth (specified in degrees)

    All args must be of equal length since we use numpy to process 

    """
    longitude1, latitude1, longitude2, latitude2 = map(np.radians, [longitude1, latitude1, longitude2, latitude2])

    longdiff = longitude2 - longitude1
    latdiff = latitude2 - latitude1

    a = np.sin(latdiff/2.0)**2 + np.cos(latitude1) * np.cos(latitude2) * np.sin(longdiff/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    haversine_dist = 6367 * c
    return haversine_dist

def euclid_distance(longitude1, latitude1, longitude2, latitude2):  
  """
  returns euclidean distance
  """
  euclid = np.sqrt((longitude1 - longitude2)**2 + (latitude1 - latitude2)**2)  
  return euclid

def merge_demo_loc_vendor(demographics_df,location_df,vendors_df,is_target_present="Y"):
  '''
  1. Merge demographics and location data

  2. Then merge vendors data

  3. Then merge the target
  '''
  # Merge demographics and location data
  #location_demograph = location_df.merge(demographics_df,left_on="customer_id",right_on="akeed_customer_id",how="left")
  location_demograph = location_df.merge(demographics_df,left_on="customer_id",right_on="akeed_customer_id",how="inner")

  location_demograph.drop(columns=["akeed_customer_id"],inplace=True)

  # Fill for those customer-locations whose customer details are not available in customer-demographics data
  location_demograph['month_customer_updated'].fillna(99,inplace=True)
  location_demograph['month_customer_created'].fillna(99,inplace=True)
  location_demograph['year_customer_created'].fillna(9999,inplace=True)
  location_demograph['year_customer_updated'].fillna(9999,inplace=True)
  location_demograph['verified'].fillna(99,inplace=True)
  location_demograph["language"].fillna('null',inplace=True)
  location_demograph.fillna("null",inplace=True) 
  

  # Merge vendor data with location demograph file
  vendors_df["key"] = 1
  location_demograph["key"] = 1
  cust_vendorval = location_demograph.merge(vendors_df,on="key") #cartesian join
  cust_vendorval["CID X LOC_NUM X VENDOR"] = cust_vendorval["customer_id"] + " X " + cust_vendorval["location_number"].astype('str') + " X " + cust_vendorval["vendor_id"].astype('str')
  cust_vendorval = cust_vendorval.iloc[0:1000000]
  
  # Compute haversine distance between vendor and customer
  # Extract features comparing haversine distance and serving distance of the vendor using differences and ratios
  cust_vendorval['haversine_distance'] = haversine_np(cust_vendorval['longitude_x'],cust_vendorval['latitude_x'],cust_vendorval['longitude_y'],cust_vendorval['latitude_y'])
  cust_vendorval['euclid_distance']    = euclid_distance(cust_vendorval['longitude_x'],cust_vendorval['latitude_x'],cust_vendorval['longitude_y'],cust_vendorval['latitude_y'])
  cust_vendorval['distance_diff']      = cust_vendorval['haversine_distance'] - cust_vendorval['serving_distance']
  cust_vendorval['distance_ratio']     = cust_vendorval['haversine_distance'] / cust_vendorval['serving_distance']
  cust_vendorval['latitude_diff']      = cust_vendorval["latitude_x"] - cust_vendorval["latitude_y"]
  cust_vendorval['longitude_diff']     = cust_vendorval["longitude_x"] - cust_vendorval["longitude_y"]
  cust_vendorval['latitude_sum']      = cust_vendorval["latitude_x"] + cust_vendorval["latitude_y"]
  cust_vendorval['longitude_sum']     = cust_vendorval["longitude_x"] + cust_vendorval["longitude_y"]

  if is_target_present:
    # Create target variable
    # Merge feature data with order data to create the target variable
    # Those with matches in the order table are considered as positive class since the customer has ordered from the vendor
    orders = pd.read_csv(input_location + orders_file)
    orders.drop_duplicates(subset=['CID X LOC_NUM X VENDOR'],inplace=True)
    orders["target"] = 1   
    cust_vendorval = pd.merge(cust_vendorval,orders[['CID X LOC_NUM X VENDOR','target']],on='CID X LOC_NUM X VENDOR',how="left")
    cust_vendorval["target"].fillna(0,inplace=True)
    
  return cust_vendorval

def get_features_label(customer_demo_file,customer_locn_file,is_target_present="Y"):
    """
    This method accepts the file names for customer-demographics and customer-locations to create the features and labels
    """

    vendors = process_vendors(input_location,vendor_file)
    vendor_summary = process_vendor_summary(input_location,orders_file)
    vendors = pd.merge(vendors,vendor_summary,on=["vendor_id"],how="inner")

    customer_demo = process_customer_demo(input_location,customer_demo_file)
    customer_locn = process_customer_location(input_location,customer_locn_file)
    cust_vendor_features  = merge_demo_loc_vendor(customer_demo,customer_locn,vendors,is_target_present)    

    if is_target_present:
      cust_vendor_features = cust_vendor_features[categorical_features + numerical_features + ["CID X LOC_NUM X VENDOR"] + ["target"]]
    else:
      cust_vendor_features = cust_vendor_features[categorical_features + numerical_features  + ["CID X LOC_NUM X VENDOR"]]

    return cust_vendor_features

def build_model():  
  tf.keras.backend.clear_session()

  #Embedding layers to process the categorical variables
  monvencre_input = Input(shape=(1),name="month_vendor_created")
  monvencre_embedding = Embedding(monvencre_count+1, monvencre_embed_dim, input_length=1,name="Emb_month_vendor_created")(monvencre_input)
  flatten_monvencre = Flatten()(monvencre_embedding)


  custcity_input = Input(shape=(1),name="customer_city")
  custcity_embedding = Embedding(custcity_count+1, custcity_embed_dim, input_length=1,name="Emb_Customer_City")(custcity_input)
  flatten_custcity = Flatten()(custcity_embedding)

  custcountry_input = Input(shape=(1),name="customer_country")
  custcountry_embedding = Embedding(custcountry_count+1, custcountry_embed_dim, input_length=1,name="Emb_Customer_Country")(custcountry_input)
  flatten_custcountry = Flatten()(custcountry_embedding)

  primarytag_input = Input(shape=(1),name="primary_tag")
  primarytag_embedding = Embedding(primarytag_count+1, primarytag_embed_dim, input_length=1,name="Emb_PrimaryTag")(primarytag_input)
  flatten_primarytag = Flatten()(primarytag_embedding)

  vendortag_input = Input(shape=(max_foodcnt),name="vendor_tags")
  vendortag_embedding = Embedding(vendortag_count+1, vendortag_embed_dim, input_length=max_foodcnt,name="Emb_Vendor_Tags")(vendortag_input)
  flatten_vendortag = Flatten()(vendortag_embedding)

  #Feed forward network to process the numerical and one hot encoded data
  numerical_input = Input(shape=(num_ohe_dim),name="numerical_inputs")
  dense_num = Dense(60,activation = 'relu', name = 'dense_layer_for_numbers')(numerical_input)

  concatenate = Concatenate()([flatten_monvencre,flatten_custcity,flatten_custcountry,flatten_primarytag,flatten_vendortag,dense_num])

  dense1 = Dense(30,activation = 'relu', name = 'dense_layer1_after_concat')(concatenate) 
  dropout1 = Dropout(0.0,name="dropout1")(dense1)

  dense2 = Dense(30,activation = 'relu', name = 'dense_layer2_after_concat')(dropout1)
  dropout2 = Dropout(0.0,name="dropout2")(dense2)

  dense3 = Dense(30,activation = 'relu', name = 'dense_layer3_after_concat')(dropout2)

  output_layer = Dense(2,activation = 'softmax',name = 'Output')(dense3)
  model = Model(inputs=[monvencre_input,custcity_input,custcountry_input,primarytag_input,vendortag_input,numerical_input],outputs=output_layer)

  return model



def get_vendor_data(location,vendor_file):
  vendors = process_vendors(new_location,vendor_file)
  vendor_summary = process_vendor_summary(new_location,orders_file)
  vendors = pd.merge(vendors,vendor_summary,on=["vendor_id"],how="inner")
  return vendors


def final_pred(customer_demographics,customer_locations):

  '''
  This function accepts the demographics and locations file names containing the raw features of the customers
  Vendor files are the same between training and test customers

  Returns:
  Predictions of the customers for all 100 vendors
  Metrics - Entropy Loss, Combined Recall, Precision, Class-Weighted Average F1, Equally-Weighted Average F1

  '''
  status_upd = st.text("Recommendations Kickstarted!!")
  #Using the raw features from the files, derives the features or useful information to train the model
  X = get_features_label(customer_demographics,customer_locations,is_target_present="N")
  X_key = X.pop('CID X LOC_NUM X VENDOR')
  
  status_upd.text("Feature Generation Completed!!")
  #Makes the extracted features model ready using the trained Encoders, Scalers and Tokenizers
  X_cat_ohe = ohe.transform(X[ohe_features]).toarray()
  X_numcols_minmax = minmaxscaler.fit_transform(X[numerical_features].to_numpy())
  X_ohe_num = np.hstack([X_numcols_minmax,X_cat_ohe])
  X_monvencre_seq = np.array(monvencre_tokenizer.texts_to_sequences(X["month_vendor_created"].values))
  X_custcity_seq = np.array(custcity_tokenizer.texts_to_sequences(X["customer_city"].values))
  X_custcountry_seq = np.array(custcountry_tokenizer.texts_to_sequences(X["customer_country"].values))
  X_primarytag_seq = np.array(primarytag_tokenizer.texts_to_sequences(X["primary_tags_mod"].values))
  X_vendortag_seq = vendortag_tokenizer.texts_to_sequences(X["vendor_tag_name_cleaned"].values)
  X_vendortag_padded = pad_sequences(X_vendortag_seq, padding='post', truncating='post', maxlen=max_foodcnt)
  num_ohe_dim = X_ohe_num.shape[1]
  status_upd.text("Feature Scaling and Tokenization Completed!!")
  #Load the trained model
  reco_model = keras.models.load_model(model_location+"dl_model")
  status_upd.text("Models Loaded!!")
  #Predict using the trained model
  y_pred = reco_model.predict([X_monvencre_seq,X_custcity_seq,X_custcountry_seq,X_primarytag_seq,X_vendortag_padded,X_ohe_num], batch_size=256)
  status_upd.text("Recommendations Given By Model!!")
  y_final = np.argmax(y_pred,axis=1)

  pred_df = pd.DataFrame(y_final,columns=["y_pred"])
  pred_df['CID X LOC_NUM X VENDOR'] = X_key
  pred_df = pred_df[["CID X LOC_NUM X VENDOR", "y_pred"]]

  status_upd.text("")
  return pred_df


def final_pred_evaluate(customer_demographics,customer_locations):

  '''
  This function accepts the demographics and locations file names containing the raw features of the customers
  Vendor files are the same between training and test customers

  Returns:
  Predictions of the customers for all 100 vendors
  Metrics - Entropy Loss, Combined Recall, Precision, Class-Weighted Average F1, Equally-Weighted Average F1

  '''
  status_upd = st.text("Recommendations Kickstarted!!")
  #Using the raw features from the files, derives the features or useful information to train the model
  X = get_features_label(customer_demographics,customer_locations,is_target_present="Y")  
  X_key = X.pop('CID X LOC_NUM X VENDOR')
  y = X.pop('target') 
  
  
  y = tf.keras.utils.to_categorical(y, 2)
  status_upd.text("Feature Generation Completed!!")
  #Makes the extracted features model ready using the trained Encoders, Scalers and Tokenizers
  X_cat_ohe = ohe.transform(X[ohe_features]).toarray()
  X_numcols_minmax = minmaxscaler.fit_transform(X[numerical_features].to_numpy())
  X_ohe_num = np.hstack([X_numcols_minmax,X_cat_ohe])
  X_monvencre_seq = np.array(monvencre_tokenizer.texts_to_sequences(X["month_vendor_created"].values))
  X_custcity_seq = np.array(custcity_tokenizer.texts_to_sequences(X["customer_city"].values))
  X_custcountry_seq = np.array(custcountry_tokenizer.texts_to_sequences(X["customer_country"].values))
  X_primarytag_seq = np.array(primarytag_tokenizer.texts_to_sequences(X["primary_tags_mod"].values))
  X_vendortag_seq = vendortag_tokenizer.texts_to_sequences(X["vendor_tag_name_cleaned"].values)
  X_vendortag_padded = pad_sequences(X_vendortag_seq, padding='post', truncating='post', maxlen=max_foodcnt)
  num_ohe_dim = X_ohe_num.shape[1]
  status_upd.text("Feature Scaling and Tokenization Completed!!")
  #Load the trained model
  reco_model = keras.models.load_model(model_location+"dl_model")
  status_upd.text("Models Loaded!!")
  #Predict using the trained model
  y_pred = reco_model.predict([X_monvencre_seq,X_custcity_seq,X_custcountry_seq,X_primarytag_seq,X_vendortag_padded,X_ohe_num], batch_size=256)
  status_upd.text("Recommendations Given By Model!!")
  y_final = np.argmax(y_pred,axis=1)
  y_true =  np.argmax(y,axis=1)

  pred_df = pd.DataFrame(y_final,columns=["y_pred"])
  pred_df['y_true'] = y_true
  pred_df['CID X LOC_NUM X VENDOR'] = X_key
  pred_df = pred_df[["CID X LOC_NUM X VENDOR", "y_pred","y_true"]]

  class_avgF1 = f1_score(pred_df['y_true'].values, pred_df['y_pred'].values,average='macro')

  status_upd.text("Scores Obtained")
  status_upd.text("")
  return class_avgF1,pred_df


def main():

  st.header("Restaurant Recommendation Case Study")
  menu = ["None","Perform EDA","Get Recommendations & its F1 score","Get Only Recommendations"]
  main_choice = st.sidebar.selectbox("Menu",menu)

  if main_choice == "Perform EDA":

    eda_choice = st.radio("What do you want to do?",
     ('Vendor Details', 'Vendor Summary from Orders','Vendor Locations','Customer Details', 'Customer Locations'))

    if eda_choice == 'Vendor Details':
        st.subheader("Processed Vendor Details")
        load_status = st.text("Please wait...Loading")
        vendors = process_vendors(new_location,vendor_file)
        st.dataframe(vendors)
        load_status.text("Loaded!!")
        load_status.text("")

    elif eda_choice == 'Vendor Summary from Orders':
       st.subheader("Vendor Summary from Orders")
       load_status = st.text("Please wait...Loading")
       vendor_summary = process_vendor_summary(new_location,orders_file)
       st.dataframe(vendor_summary)
       load_status.text("Loaded!!")
       load_status.text("")

    elif eda_choice == 'Vendor Locations':
        st.subheader("Vendor Locations")
        load_status = st.text("Please wait...Loading")
        vendors = process_vendors(new_location,vendor_file)
        locations = vendors[(((vendors["latitude"] >= -90.0) & (vendors["latitude"] <= 90.0)) & ((vendors["longitude"] >= -180.0) & (vendors["longitude"] <= 180.0)))][['latitude','longitude']]
        st.map(locations)
        load_status.text("Loaded!!")
        load_status.text("")

    elif eda_choice == 'Customer Details':
        st.subheader("Processed Customer Details")
        load_status = st.text("Please wait...Loading")
        customer_demo = process_customer_demo(new_location,train_customer_demographics)
        st.dataframe(customer_demo)
        load_status.text("Loaded!!")
        load_status.text("")

    elif eda_choice == 'Customer Locations':
        st.subheader("Customer Locations")
        load_status = st.text("Please wait...Loading")
        cust_locn = process_customer_location(new_location,train_customer_locations)
        cust_lo = cust_locn[(((cust_locn["latitude"] >= -90.0) & (cust_locn["latitude"] <= 90.0)) & ((cust_locn["longitude"] >= -180.0) & (cust_locn["longitude"] <= 180.0)))][['latitude','longitude']]
        st.map(cust_lo)
        load_status.text("Loaded!!")
        load_status.text("")


  elif main_choice == "Get Recommendations & its F1 score":

    load_status = st.text("Generating Recommendations.....")
    f1score, train_pred = final_pred_evaluate(train_customer_demographics,train_customer_locations)
    f1score = round(f1score,2)
    st.metric("F1 score of the recommendations ",f1score)
    st.dataframe(train_pred.iloc[0:100,:])
    load_status.text("Loaded!!")
    load_status.text("")

  elif main_choice == "Get Only Recommendations":

    load_status = st.text("Generating Recommendations.....")
    test_pred = final_pred(test_customer_demographics,test_customer_locations)
    st.dataframe(test_pred)
    load_status.text("Loaded!!")
    load_status.text("")


input_location = r'C:/Users/Thomas/JoshiniRepo/CS1/data/'
model_location = r'C:/Users/Thomas/JoshiniRepo/CS1/'
new_location = r'C:/Users/Thomas/JoshiniRepo/CS1/data/'
scaler_location = r'C:/Users/Thomas/JoshiniRepo/CS1/scalers/'

train_customer_demographics = "train_customers.csv"
test_customer_demographics = "test_customers.csv"
train_customer_locations = "train_locations.csv"
test_customer_locations  = "test_locations.csv"
vendor_file = "vendors.csv"
orders_file = "orders.csv"
categorical_features = ['gender','language_x','location_type','customer_city','customer_country', 'verified_x','year_customer_created', 
                    'month_customer_created', 'year_customer_updated', 'month_customer_updated','language_y','vendor_city','vendor_country', 
              'vendor_category_id','delivery_charge', 'is_open', 'commission',  'status', 'verified_y', 'rank', 'device_type','primary_tags_mod', 
              'year_vendor_created', 'month_vendor_created', 'year_vendor_updated', 'month_vendor_updated','vendor_tag_name_cleaned']

ohe_features = ['gender','language_x', 'location_type', 'verified_x','year_customer_created','year_customer_updated','language_y','vendor_city','vendor_country',
'vendor_category_id','delivery_charge','is_open','commission','status','verified_y','rank','device_type','year_vendor_created',
'year_vendor_updated', 'month_vendor_updated']


numerical_features = ['latitude_x', 'longitude_x','latitude_y', 'longitude_y','open_duration','customer_id_count', 'customer_id_nunique', 
                      'serving_distance','prepration_time', 'discount_percentage', 'tag_counts', 'vendor_rating',
                'payment_mode_nunique', 'promo_code_count', 'vendor_discount_amount_sum', 'promo_code_discount_percentage_mean', 
                'item_count_median', 'grand_total_median', 'driver_rating_median', 'deliverydistance_mean', 'preparationtime_mean', 
                'order_turnaround_min', 'order_turnaround_max', 'order_turnaround_mean','haversine_distance', 'euclid_distance', 
                'distance_diff', 'distance_ratio', 'latitude_diff','longitude_diff','latitude_sum','longitude_sum']

embedded_features = ['customer_city', 'customer_country', 'primary_tags_mod', 'month_vendor_created']

max_foodcnt = 10
monvencre_embed_dim   = 2
custcity_embed_dim    = 3
custcountry_embed_dim = 2 
primarytag_embed_dim  = 3
vendortag_embed_dim   = 3

#Load trained feature encoders, scalers and tokenizers
with open(scaler_location + 'onehotencoder.pickle', 'rb') as handle:
  ohe = pickle.load(handle)
with open(scaler_location + 'minmaxscaler.pickle', 'rb') as handle:
  minmaxscaler = pickle.load(handle)
with open(scaler_location + 'monvencre_tokenizer.pickle', 'rb') as handle:
  monvencre_tokenizer = pickle.load(handle)
with open(scaler_location + 'custcity_tokenizer.pickle', 'rb') as handle:
  custcity_tokenizer = pickle.load(handle)
with open(scaler_location + 'custcountry_tokenizer.pickle', 'rb') as handle:
  custcountry_tokenizer = pickle.load(handle)
with open(scaler_location + 'primarytag_tokenizer.pickle', 'rb') as handle:
  primarytag_tokenizer = pickle.load(handle)
with open(scaler_location + 'vendortag_tokenizer.pickle', 'rb') as handle:
  vendortag_tokenizer = pickle.load(handle)

monvencre_count   = len(monvencre_tokenizer.word_index)
custcity_count    = len(custcity_tokenizer.word_index)
custcountry_count = len(custcountry_tokenizer.word_index)
primarytag_count  = len(primarytag_tokenizer.word_index)
vendortag_count   = len(vendortag_tokenizer.word_index)

if __name__ == '__main__':
	main()