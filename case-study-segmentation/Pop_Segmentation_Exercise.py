#!/usr/bin/env python
# coding: utf-8

# # Population Segmentation with SageMaker

# In[1]:
import pandas as pd
import numpy as np
import os
import io

import matplotlib.pyplot as plt
import matplotlib
get_ipython().magic('matplotlib inline')


# In[2]:
# sagemaker libraries
import boto3
import sagemaker

# ## Loading the Data from Amazon S3
# In[3]:
# boto3 client to get S3 data
s3_client = boto3.client('s3')
bucket_name='aws-ml-blog-sagemaker-census-segmentation'
# In[4]:
# get a list of objects in the bucket
obj_list=s3_client.list_objects(Bucket=bucket_name)

# print object(s)in S3 bucket
files=[]
for contents in obj_list['Contents']:
    files.append(contents['Key'])
    
print(files)

# In[5]:
# there is one file --> one key
file_name=files[0]
print(file_name)

# Retrieve the data file from the bucket with a call to `client.get_object()`.
# In[6]:
# get an S3 object by passing in the bucket and file name
data_object = s3_client.get_object(Bucket=bucket_name, Key=file_name)
display(data_object)

# In[7]:
# information is in the "Body" of the object
data_body = data_object["Body"].read()
print('Data type: ', type(data_body))

# In[8]:
# read in bytes data
data_stream = io.BytesIO(data_body)

# create a dataframe
counties_df = pd.read_csv(data_stream, header=0, delimiter=",") 
counties_df.head()

# ## Exploratory Data Analysis (EDA)
# In[9]:
counties_df.shape


# In[10]:
# drop any incomplete rows of data, and create a new df
clean_counties_df = counties_df.dropna(axis=0)
clean_counties_df.shape


# In[11]:
clean_counties_df.dtypes


# ### EXERCISE: Create a new DataFrame, indexed by 'State-County'
# In[12]:
# index data by 'State-County'
# clean_counties_df.index= # your code here
clean_counties_df.index = clean_counties_df.State + "-" + clean_counties_df.County


# In[13]:
# drop the old State and County columns, and the CensusId column
# clean df should be modified or created anew
clean_counties_df = clean_counties_df.drop(["State", "County", "CensusId"], axis=1)
clean_counties_df.shape

# In[14]:
# features
features_list = clean_counties_df.columns.values
print('Features: \n', features_list)


# ## Visualizing the Data
# In[15]:
# transportation (to work)
transport_list = ['Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp']
n_bins = 30 # can decrease to get a wider bin (or vice versa)

for column_name in transport_list:
    ax=plt.subplots(figsize=(6,3))
    # get data by column_name and display a histogram
    ax = plt.hist(clean_counties_df[column_name], bins=n_bins)
    title="Histogram of " + column_name
    plt.title(title, fontsize=12)
    plt.show()

# ### EXERCISE: Create histograms of your own

# In[16]:
# create a list of features that you want to compare or examine
my_list = ['Employed','SelfEmployed','FamilyWork','Unemployment']

# histogram creation code is similar to above
for column_name in my_list:
    ax=plt.subplots(figsize=(6,3))
    # get data by column_name and display a histogram
    ax = plt.hist(clean_counties_df[column_name])
    title="Histogram of " + column_name
    plt.title(title, fontsize=12)
    plt.show()


# ### EXERCISE: Normalize the data
# In[17]:
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

counties_scaled = pd.DataFrame(scaler.fit_transform(clean_counties_df.astype(float)))
counties_scaled.columns = clean_counties_df.columns
counties_scaled.index = clean_counties_df.index
counties_scaled.head()


# # Data Modeling
# In[18]:
from sagemaker import get_execution_role

session = sagemaker.Session() # store the current SageMaker session

# get IAM role
role = get_execution_role()
print(role)


# In[19]:
# get default bucket
bucket_name = session.default_bucket()
print(bucket_name)

# ## Define a PCA Model
# In[20]:
# define location to store model artifacts
prefix = 'counties'

output_path='s3://{}/{}/'.format(bucket_name, prefix)
print('Training artifacts will be uploaded to: {}'.format(output_path))


# In[21]:
# define a PCA model
from sagemaker import PCA

# this is current features - 1
# you'll select only a portion of these to use, later
N_COMPONENTS = 33

pca_SM = PCA(
    role=role,
    train_instance_count=1,
    train_instance_type='ml.c4.xlarge',
    output_path=output_path, #specified, above
    num_components=N_COMPONENTS, 
    sagemaker_session=session
)

# ### Convert data into a RecordSet format
# In[22]:
# convert df to np array
train_data_np = counties_scaled.values.astype('float32')

# convert to RecordSet format
formatted_train_data = pca_SM.record_set(train_data_np)


# ## Train the model
# In[23]:
get_ipython().run_cell_magic('time', '', '\n# train the PCA mode on the formatted data\npca_SM.fit(formatted_train_data)')


# ## Accessing the PCA Model Attributes
# In[24]:
# Get the name of the training job, it's suggested that you copy-paste
# from the notebook or from a specific job in the AWS Console
training_job_name='pca-2021-04-20-02-23-11-231'

# where the model is saved, by default
model_key = os.path.join(prefix, training_job_name, 'output/model.tar.gz')
print(model_key)

# download and unzip model
boto3.resource('s3').Bucket(bucket_name).download_file(model_key, 'model.tar.gz')

# unzipping as model_algo-1
os.system('tar -zxvf model.tar.gz')
os.system('unzip model_algo-1')


# ### MXNet Array
# In[25]:
get_ipython().system('pip install mxnet')

# In[26]:
import mxnet as mx
# loading the unzipped artifacts
pca_model_params = mx.ndarray.load('model_algo-1')

# what are the params
print(pca_model_params)

# ## PCA Model Attributes
# In[27]:
# get selected params
s = pd.DataFrame(pca_model_params['s'].asnumpy())
v = pd.DataFrame(pca_model_params['v'].asnumpy())

# ## Data Variance
# In[28]:
# looking at top 5 components
n_principal_components = 5

start_idx = N_COMPONENTS - n_principal_components  # 33-n

# print a selection of s
print(s.iloc[start_idx:, :])


# ### EXERCISE: Calculate the explained variance
# In[29]:
# Calculate the explained variance for the top n principal components
# you may assume you have access to the global var N_COMPONENTS
def explained_variance(s, n_top_components):
    '''Calculates the approx. data variance that n_top_components captures.
       :param s: A dataframe of singular values for top components; 
           the top value is in the last row.
       :param n_top_components: An integer, the number of top components to use.
       :return: The expected data variance covered by the n_top_components.'''
    
    # your code here
    start_idx = s.shape[0] - n_top_components
    var_exp = np.square(s.iloc[start_idx:, :]).sum() / np.square(s).sum()
    return var_exp[0]


# ### Test Cell
# In[30]:
# test cell
n_top_components = 7 #TODO: select a value for the number of top components

# calculate the explained variance
exp_variance = explained_variance(s, n_top_components)
print('Explained Variance: %.4f' % exp_variance)

# In[31]:
# features
features_list = counties_scaled.columns.values
print('Features: \n', features_list[:n_top_components])

# ## Component Makeup
# In[32]:
import seaborn as sns

def display_component(v, features_list, component_num, n_weights=10):
    
    # get index of component (last row - component_num)
    row_idx = N_COMPONENTS-component_num

    # get the list of weights from a row in v, dataframe
    v_1_row = v.iloc[:, row_idx]
    v_1 = np.squeeze(v_1_row.values)

    # match weights to features in counties_scaled dataframe, using list comporehension
    comps = pd.DataFrame(list(zip(v_1, features_list)), 
                         columns=['weights', 'features'])

    # we'll want to sort by the largest n_weights
    # weights can be neg/pos and we'll sort by magnitude
    comps['abs_weights']=comps['weights'].apply(lambda x: np.abs(x))
    sorted_weight_data = comps.sort_values('abs_weights', ascending=False).head(n_weights)

    # display using seaborn
    ax=plt.subplots(figsize=(10,6))
    ax=sns.barplot(data=sorted_weight_data, 
                   x="weights", 
                   y="features", 
                   palette="Blues_d")
    ax.set_title("PCA Component Makeup, Component #" + str(component_num))
    plt.show()


# In[33]:
# display makeup of first component
num = 1
display_component(v, counties_scaled.columns.values, component_num=num, n_weights=10)

# # Deploying the PCA Model
# In[34]:
get_ipython().run_cell_magic('time', '', "# this takes a little while, around 7mins\npca_predictor = pca_SM.deploy(initial_instance_count=1,\n                              instance_type='ml.t2.medium')")

# In[35]:
# pass np train data to the PCA model
train_pca = pca_predictor.predict(train_data_np)


# In[36]:
# check out the first item in the produced training features
print(train_pca[0])

# ### EXERCISE: Create a transformed DataFrame

# In[45]:
# create dimensionality-reduced data
def create_transformed_df(train_pca, counties_scaled, n_top_components):
    """
    Return a dataframe of data points with component features. 
    The dataframe should be indexed by State-County and contain component values.
    :param train_pca: A list of pca training data, returned by a PCA model.
    :param counties_scaled: A dataframe of normalized, original features.
    :param n_top_components: An integer, the number of top components to use.
    :return: A dataframe, indexed by State-County, with n_top_component values as columns. 
    """
    # create a dataframe of component features, indexed by State-County
    # your code here
    
    ## Extract PCs
    df_transformed = pd.DataFrame()
    for data in train_pca:
        pcs = data.label['projection'].float32_tensor.values
        df_transformed = df_transformed.append([list(pcs)])
        
    df_transformed.index = counties_scaled.index
    
    ## Select & order PCs from most variant
    start_idx = counties_scaled.shape[1] - n_top_components - 1
    df_transformed = df_transformed.iloc[:, start_idx:]
    return df_transformed.iloc[:, ::-1]

# In[46]:
## Specify top n
top_n = 7

# call your function and create a new dataframe
counties_transformed = create_transformed_df(train_pca, counties_scaled, n_top_components=top_n)
counties_transformed.head()

# In[47]:
## TODO: Add descriptive column names
counties_transformed.columns = ["pc_"+str(i) for i in range(1, top_n+1)]
counties_transformed.head()

# ### Delete the Endpoint!
# In[48]:
# delete predictor endpoint
session.delete_endpoint(pca_predictor.endpoint)

# ---
# # Population Segmentation 
# ### EXERCISE: Define a k-means model
# In[49]:
# define a KMeans estimator
from sagemaker import KMeans

NUM_CLUSTER = 8

kmeans = KMeans(
    role = role,
    train_instance_count=1,
    train_instance_type='ml.c4.xlarge',
    output_path=output_path,
    k = NUM_CLUSTER
)

# ### EXERCISE: Create formatted, k-means training data
# In[50]:
# convert the transformed dataframe into record_set data
data4kmeans = counties_transformed.values.astype('float32')
data4kmeans = kmeans.record_set(data4kmeans)


# ### EXERCISE: Train the k-means model
# In[51]:
get_ipython().run_cell_magic('time', '', '# train kmeans\nkmeans.fit(data4kmeans)')

# ### EXERCISE: Deploy the k-means model
# In[52]:
get_ipython().run_cell_magic('time', '', "# deploy the model to create a predictor\nkmeans_predictor = kmeans.deploy(initial_instance_count=1,\n                                 instance_type='ml.t2.medium')")


# ### EXERCISE: Pass in the training data and assign predicted cluster labels
# In[54]:
# get the predicted clusters for all the kmeans training data
cluster_info = kmeans_predictor.predict(counties_transformed.values.astype('float32'))

# ## Exploring the resultant clusters
# In[55]:
# print cluster info for first data point
data_idx = 0

print('County is: ', counties_transformed.index[data_idx])
print()
print(cluster_info[data_idx])


# ### Visualize the distribution of data over clusters
# In[56]:
# get all cluster labels
cluster_labels = [c.label['closest_cluster'].float32_tensor.values[0] for c in cluster_info]


# In[57]:
# count up the points in each cluster
cluster_df = pd.DataFrame(cluster_labels)[0].value_counts()
print(cluster_df)

# ### Delete the Endpoint!
# In[58]:
# delete kmeans endpoint
session.delete_endpoint(kmeans_predictor.endpoint)

# ---
# # Model Attributes & Explainability
# ### EXERCISE: Access the k-means model attributes
# In[59]:
# download and unzip the kmeans model file
kmeans_job_name = 'kmeans-2021-04-20-03-27-33-853'

model_key = os.path.join(prefix, kmeans_job_name, 'output/model.tar.gz')

# download the model file
boto3.resource('s3').Bucket(bucket_name).download_file(model_key, 'model.tar.gz')
os.system('tar -zxvf model.tar.gz')
os.system('unzip model_algo-1')


# In[60]:
# get the trained kmeans params using mxnet
kmeans_model_params = mx.ndarray.load('model_algo-1')

print(kmeans_model_params)

# In[61]:
# get all the centroids
cluster_centroids=pd.DataFrame(kmeans_model_params[0].asnumpy())
cluster_centroids.columns=counties_transformed.columns

display(cluster_centroids)


# ### Visualizing Centroids in Component Space
# In[62]:
# generate a heatmap in component space, using the seaborn library
plt.figure(figsize = (12,9))
ax = sns.heatmap(cluster_centroids.T, cmap = 'YlGnBu')
ax.set_xlabel("Cluster")
plt.yticks(fontsize = 16)
plt.xticks(fontsize = 16)
ax.set_title("Attribute Value by Centroid")
plt.show()

# In[63]:
component_num=7
display_component(v, counties_scaled.columns.values, component_num=component_num)


# ### Natural Groupings
# In[64]:
# add a 'labels' column to the dataframe
counties_transformed['labels']=list(map(int, cluster_labels))

# sort by cluster label 0-6
sorted_counties = counties_transformed.sort_values('labels', ascending=True)
# view some pts in cluster 0
sorted_counties.head(20)

# In[65]:
# get all counties with label == 1
cluster=counties_transformed[counties_transformed['labels']==1]
cluster.head()


# In[72]:
fig, ax = plt.subplots(figsize=(8,8))
scatter = ax.scatter(
    counties_transformed["pc_1"],
    counties_transformed["pc_2"],
    c = counties_transformed["labels"]
)

legend1 = ax.legend(*scatter.legend_elements(), loc="lower right", title="Cluster")
ax.add_artist(legend1)

handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show()