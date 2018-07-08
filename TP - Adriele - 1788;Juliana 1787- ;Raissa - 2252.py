
# coding: utf-8

# In[34]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
import matplotlib as mpl
import seaborn as sns
from ipywidgets import interact
from pylab import *
import matplotlib.colors
from wordcloud import WordCloud, STOPWORDS


# In[35]:


#Ler o dataset
musicas = pd.read_csv('data.csv', index_col=False, squeeze=True, low_memory=False);
musicas.rename(columns={'Track Name':'TrackName'}, inplace=True)
musicas


# In[36]:


#Verificar se há alguma célula com valor negativo
musicas[(musicas['Streams'] < 0) | (musicas['Position'] < 0)]


# In[37]:


#Eliminar valores negativos
musicas = musicas[(musicas['Streams'] > 0) & (musicas['Position'] > 0)]


# In[38]:


# Verifica se há células nulas
result1 = len(musicas) - pd.isnull(musicas.Position).count()
result2 = len(musicas) - pd.isnull(musicas.TrackName).count()
result3 = len(musicas) - pd.isnull(musicas.Artist).count()
result4 = len(musicas) - pd.isnull(musicas.Streams).count()
result5 = len(musicas) - pd.isnull(musicas.URL).count()
result6 = len(musicas) - pd.isnull(musicas.Date).count()
result7 = len(musicas) - pd.isnull(musicas.Region).count()

resultado = result1 + result2 + result3 + result4 + result5 + result6 + result7
print("Quantidade de celulas nulas: ", resultado)


# In[39]:


# Eliminar células com campos vazios
musicas.dropna(axis=0, subset=['Position'], inplace=True)
musicas.dropna(axis=0, subset=['TrackName'], inplace=True)
musicas.dropna(axis=0, subset=['Artist'], inplace=True)
musicas.dropna(axis=0, subset=['Streams'], inplace=True)
musicas.dropna(axis=0, subset=['URL'], inplace=True)
musicas.dropna(axis=0, subset=['Date'], inplace=True)
musicas.dropna(axis=0, subset=['Region'], inplace=True)

musicas


# In[40]:


#Eliminar ruídos nos nomes das músicas e artistas
def corrigir (nome):
    nome = nome.replace('#', '').replace('$', 's').replace('*', '')
    return nome

musicas.Artist = musicas.Artist.apply(corrigir)
musicas.TrackName = musicas.TrackName.apply(corrigir)

musicas


# In[41]:


# Tira os espaços em branco do começo e fim da String, para evitar que as mesmas sejam diferenciadas

musicas['TrackName'] = musicas['TrackName'].str.strip()
musicas['Artist'] = musicas['Artist'].str.strip()
musicas['URL'] = musicas['URL'].str.strip()
musicas['Date'] = musicas['Date'].str.strip()
musicas['Region'] = musicas['Region'].str.strip()


# In[42]:


# Tranforma as datas em dias transcorridos para facilitar na manipulação

print ("Primeira Data: {}\nÚltima Data: {}".format(musicas.Date.min(),musicas.Date.max())) # Verifica primeira e última data
Dates = pd.to_datetime(musicas.Date) # Verifica o formato da data
Days = Dates.sub(Dates[0], axis = 0) # Subtrai resultados redundantes
Days = Days / np.timedelta64(1, 'D') # converte para Float
print ("Primeiro Dia: {}\nÚltimo Dia: {}".format(Days.min(), Days.max())) # check converted first and last days elapsed
musicas['Days'] = Days # Adiciona a nova coluna com Float's ao dataframe
# musicas.drop('Date', axis = 1, inplace = True)


# In[43]:


musicas['Year'] = musicas.Date.str[:4]
musicas['Month'] = musicas.Date.str[5:7]


# In[44]:


# Cálculos estatísticos com base na coluna "Streams" utilizando a função describe()

musicas['Streams'].describe()


# In[45]:


#Verificar a quantidade de músicas distintas no dataset
Nomes = musicas['TrackName']
totalNomes = len(Nomes.value_counts())
print( "O total de nome de musicas distintas é de : ", totalNomes)
print("E o tamanho do nosso dataset é de ", len(musicas), "linhas (musicas)")


# In[46]:


#Quantidade de Streams por região
paises = musicas.groupby('Region')
paises_sum = paises.sum()
paises_sum['Streams'].plot(kind='bar')
plt.yscale('log')
plt.rcParams['figure.figsize'] = (18,8)
plt.legend(loc='upper right', prop={'size':12}, fontsize=1)
plt.show()


# In[47]:


#Quantidade de Streams por Dia/Data

dias = musicas.groupby(['Days', 'Date'])
dias_sum = dias.sum()

dias_sum


# In[48]:


#Quantidade de Streams por mês
meses = musicas.groupby(['Year', 'Month'])
meses_sum = meses.sum()
meses_sum


# In[57]:


#Ranking dos meses com mais Streams
fig = gcf()
plt.style.use('seaborn-whitegrid')

meses_sum['Streams'].plot(kind='bar')
# plt.rcParams['figure.figsize'] = (4,2)
plt.legend(loc='upper right', prop={'size':10}, fontsize=5)
plt.yscale('log')
plt.savefig('mesesMaisStreams.png') 
plt.show()


# In[50]:


#Ranking dos 25 artistas mais acessados
artistas = musicas.groupby('Artist')
artistas_soma = artistas.Streams.sum()

#Cria um novo dataframe para manipulação
contStreams=pd.DataFrame(artistas_soma)
contStreamsM = contStreams.Streams.sort_values(ascending=False)
contStreamsM = contStreamsM[:25]

contStreamsM.plot(kind='bar')
plt.rcParams['figure.figsize'] = (18,8)
plt.legend(loc='upper right', prop={'size':12}, fontsize=1)
plt.yscale('log')
plt.show()


# In[62]:


#Ranking das 25 músicas mais acessadas
musicas_mais = musicas.groupby('TrackName')
musicas_mais_soma = musicas_mais.Streams.sum()

#Cria um novo dataframe para manipulação
contMusics=pd.DataFrame(musicas_mais_soma)
contMusicsM = contMusics.Streams.sort_values(ascending=False)
contMusicsM = contMusicsM[:25]

contMusicsM.plot(kind='bar')
plt.rcParams['figure.figsize'] = (30,12)
plt.legend(loc='upper right', prop={'size':12}, fontsize=1)
plt.yscale('log')
plt.show()
plt.savefig('musicas.png') 


# In[ ]:


fig = gcf()
ax = fig.gca()
plt.style.use('seaborn-poster')

y = musicas['Streams']
x = musicas['Date'] 

plt.title('Streams por Data', fontsize='14')
plt.xlabel('Data', fontsize='14')
plt.ylabel('Streams', fontsize='14')


#plt.rcParams['figure.figsize'] = (100,18)
#plt.plot(x, y ,'o',color='blue');
plt.scatter(x, y ,color='blue',s=50, alpha=.5 );

xmarks=['2017-01-01','2017-03-18','2017-08-30','2018-01-09'] # 5 registros
plt.xticks(xmarks)

labels = [item.get_text() for item in ax.get_xticklabels()]
labels[0] = '2017-01-01'
labels[1] = '2017-03-18'
labels[2] = '2017-08-30'
labels[3] = '2018-01-09'
ax.set_xticklabels(labels)

plt.savefig('StreamsPorData.png') 


# In[52]:


#Ranking dos 50 musicas mais acessadas, com seus artistas
artistas_mus = musicas.groupby(['Artist', 'TrackName'])
artistas_mus = artistas_mus.Streams.sum()
artistas_mus
#Cria um novo dataframe para manipulação
artistas_musf=pd.DataFrame(artistas_mus)
artistas_musf = artistas_musf.Streams.sort_values(ascending=False)
artistas_musf = artistas_musf[:50]
art = artistas_musf.reset_index()
art


# In[54]:


fig = gcf()
ax = fig.gca()
plt.style.use('seaborn-poster')

X = art['Artist']
Y = art['TrackName']
C = art['Streams']

plt.rcParams['figure.figsize'] = (80,25)

cmap = mpl.colors.ListedColormap([ 'blue', 'orange',  'red'])


s = ax.scatter(X,Y,c=C,lw=0.3, s=500, cmap=cmap) #plasma,hsv  'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'autumn']

plt.title('Gráfico de artistas por Track name em função das streams.', fontsize='50')
plt.ylabel('Track Name', fontsize='50')
plt.xlabel('Artist', fontsize='50')


#norm = mpl.colors.Normalize(vmin=1, vmax=20)
cb = plt.colorbar(s)

cb.set_label('Streams', fontsize='50')


plt.savefig('ArtistasPorStreamsCores.png') 


# In[56]:


teste=pd.DataFrame(artistas_mus)
teste = teste.Streams.sort_values(ascending=False)
teste = teste.reset_index()
teste

comment_words = ' '
for val in teste.Artist:
    comment_words = comment_words + val + ' '
    
stopwords = set(STOPWORDS)
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)

# plot the WordCloud image                       
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.savefig('wordcloud.png') 
plt.show()


# In[21]:


dados = paises_sum['Streams']
dados = pd.DataFrame(data=dados)
paises_sum


# In[22]:


nomes_paises_aux = musicas.drop(['Position', 'TrackName', 'Streams', 'URL', 'Date', 'Days', 'Year', 'Month', 'Artist'], axis=1)
nomes_paises_aux = unique(nomes_paises_aux)
nomes_paises = pd.DataFrame(data=nomes_paises_aux)
nomes_paises


# In[23]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()

region_integer = enc.fit_transform(nomes_paises[0])
nomes_paises['RegionInteger'] = region_integer
#nomes_paises['nomes2'] = pd.to_numeric(nomes_paises[0], errors='coerce')
nomes_paises


# In[24]:


nomes_paises_teste = nomes_paises
nomes_paises_teste['Streams'] = unique(paises_sum['Streams'])


# In[25]:


from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
dataset_array = nomes_paises_teste.values
region_integer = region_integer.reshape(len(region_integer),1)
novas_features = ohe.fit_transform(region_integer)

nomes_paises = np.concatenate([nomes_paises_teste, novas_features.toarray()], axis=1)
nomes_paises.shape
dataf = pd.DataFrame(nomes_paises_teste)
dataf.head()


# In[30]:


from sklearn.model_selection import train_test_split


X_train,X_test,Y_train,Y_test = train_test_split(
    nomes_paises_teste[u'RegionInteger'].reshape((len(nomes_paises_teste['RegionInteger']), 1)), 
    nomes_paises_teste[u'Streams'], 
    test_size=0.2, random_state=0)

#reshape((len(paises_sum['RegionInteger']), 1))


# In[31]:


from sklearn.metrics import precision_recall_fscore_support
#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
print("Acurácia = {0}%".format(100*np.sum(knn.predict(X_test) == Y_test)/len(Y_test)))
print("(Precisão, Recall, FScore)")
precision_recall_fscore_support(Y_test, knn.predict(X_test), average=None)


# In[32]:


#Arvore de Decisão
from sklearn import tree
arv = tree.DecisionTreeClassifier()
arv = arv.fit(X_train, Y_train)
print("Acurácia = {0}%".format(100*np.sum(arv.predict(X_test) == Y_test)/len(Y_test)))
print("(Precisão, Recall, FScore)")
precision_recall_fscore_support(Y_test, arv.predict(X_test), average=None)


# In[33]:


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, Y_train)
print("Acurácia = {0}%".format(100*np.sum(clf.predict(X_test) == Y_test)/len(Y_test)))
print("(Precisão, Recall, FScore)")
precision_recall_fscore_support(Y_test, clf.predict(X_test), average=None)


# In[58]:


from sklearn import cluster
k_means = cluster.KMeans(n_clusters = 2)
k_means.fit(nomes_paises_teste[['RegionInteger','Streams']])


# In[59]:


print(k_means.cluster_centers_)
consumo_df = nomes_paises_teste.groupby('Streams')    [['RegionInteger', 'Streams']].mean()
consumo_df


# In[60]:


plt.figure(figsize=(14,7))

colormap= np.array(['red', 'lime'])

#Real
plt.subplot(1,2,1)
plt.scatter(nomes_paises_teste['RegionInteger'], nomes_paises_teste['Streams'], c=colormap, s=40)
plt.title('Classificação Real')

#K-Means
plt.subplot(1,2,2)
plt.scatter(nomes_paises_teste['RegionInteger'], nomes_paises_teste['Streams'], c=colormap[k_means.labels_], s=40)
plt.title('Classificação K-Means')
plt.savefig('kmeans.png')
plt.show()

