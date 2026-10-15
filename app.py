import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.set_page_config(page_title='Customer Segmentation | K-Means', page_icon='🛍️', layout='wide')
st.markdown('''<style>.stApp{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e)} h1{color:#a78bfa} div[data-testid="metric-container"]{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);border-radius:12px;padding:16px}</style>''', unsafe_allow_html=True)

PALETTE=['#a78bfa','#60a5fa','#34d399','#fbbf24','#f87171','#818cf8','#f472b6','#38bdf8','#4ade80','#fb923c']

def preprocess_data(df):
    d=df.copy()
    if 'Genre' in d.columns: d.rename(columns={'Genre':'Gender'},inplace=True)
    if 'CustomerID' in d.columns: d.drop(columns=['CustomerID'],inplace=True)
    if 'Gender' in d.columns: d['Gender']=d['Gender'].map({'Male':0,'Female':1}).fillna(d['Gender'])
    features=['Annual Income (k$)','Spending Score (1-100)']
    missing=[c for c in features if c not in d.columns]
    if missing: raise ValueError('Missing required columns: '+', '.join(missing))
    X=d[features].apply(pd.to_numeric,errors='coerce').dropna()
    scaler=StandardScaler(); X_scaled=scaler.fit_transform(X)
    return d,X,X_scaled,scaler

def dark_fig():
    fig,ax=plt.subplots(figsize=(8,5)); fig.patch.set_facecolor('#1e1b4b'); ax.set_facecolor('#1e1b4b')
    ax.tick_params(colors='#cbd5e1'); ax.xaxis.label.set_color('#cbd5e1'); ax.yaxis.label.set_color('#cbd5e1'); ax.title.set_color('#e2e8f0')
    for sp in ax.spines.values(): sp.set_color('#64748b')
    return fig,ax

st.title('🛍️ Customer Segmentation Dashboard')
st.caption('K-Means clustering · Exploratory data analysis · Business recommendations')
with st.sidebar:
    st.header('Configuration')
    uploaded=st.file_uploader('Upload customer CSV',type=['csv'])
    n_clusters=st.slider('Number of clusters',2,10,5)
    st.info('Use the Elbow Method and Silhouette Score to compare cluster counts.')

if uploaded is not None: df=pd.read_csv(uploaded)
else:
    path=os.path.join(os.path.dirname(__file__),'data','Mall_Customers.csv')
    if not os.path.exists(path): st.error('Upload a customer CSV to begin.'); st.stop()
    df=pd.read_csv(path)
try: processed,X,X_scaled,scaler=preprocess_data(df)
except Exception as e: st.error(str(e)); st.stop()
if len(X)<n_clusters: st.error('The dataset must contain at least as many valid rows as clusters.'); st.stop()

st.header('1. Dataset overview')
a,b,c,d=st.columns(4); a.metric('Customers',len(df)); b.metric('Features',df.shape[1]); c.metric('Missing values',int(df.isna().sum().sum()))
if 'Genre' in df.columns or 'Gender' in df.columns:
    gc='Genre' if 'Genre' in df.columns else 'Gender'; d.metric('Male customers',f"{(df[gc].astype(str).str.lower()=='male').mean()*100:.0f}%")
st.dataframe(df.head(10),use_container_width=True)
left,right=st.columns(2)
with left: st.subheader('Data types'); st.dataframe(df.dtypes.astype(str).rename('Type'),use_container_width=True)
with right: st.subheader('Missing values'); st.dataframe(df.isna().sum().rename('Missing'),use_container_width=True)

st.header('2. Exploratory data analysis')
left,right=st.columns(2)
for col,title,slot in [('Annual Income (k$)','Annual income distribution',left),('Spending Score (1-100)','Spending score distribution',right)]:
    with slot:
        fig,ax=dark_fig(); ax.hist(pd.to_numeric(df[col],errors='coerce').dropna(),bins=20,color='#a78bfa',edgecolor='#1e1b4b'); ax.set_title(title); ax.set_ylabel('Count'); st.pyplot(fig); plt.close(fig)
left,right=st.columns(2)
with left:
    fig,ax=dark_fig(); sns.heatmap(processed.select_dtypes(include='number').corr(),annot=True,fmt='.2f',cmap='coolwarm',ax=ax); ax.set_title('Correlation heatmap'); st.pyplot(fig); plt.close(fig)
with right:
    fig,ax=dark_fig(); hue='Genre' if 'Genre' in df.columns else ('Gender' if 'Gender' in df.columns else None)
    if hue:
        for i,(name,group) in enumerate(df.groupby(hue,dropna=False)):
            ax.scatter(group['Annual Income (k$)'],group['Spending Score (1-100)'],label=str(name),color=PALETTE[i%len(PALETTE)],alpha=.8)
        ax.legend()
    else: ax.scatter(X.iloc[:,0],X.iloc[:,1],color='#a78bfa')
    ax.set_xlabel('Annual Income (k$)'); ax.set_ylabel('Spending Score (1-100)'); ax.set_title('Income vs spending'); st.pyplot(fig); plt.close(fig)

st.header('3. Clustering analysis')
max_k=min(10,len(X)-1); wcss=[]
for k in range(1,max_k+1): wcss.append(KMeans(n_clusters=k,random_state=42,n_init=10).fit(X_scaled).inertia_)
model=KMeans(n_clusters=n_clusters,random_state=42,n_init=10); labels=model.fit_predict(X_scaled)
sil=silhouette_score(X_scaled,labels) if len(set(labels))>1 else 0
left,right=st.columns(2)
with left:
    fig,ax=dark_fig(); ax.plot(range(1,max_k+1),wcss,marker='o',color='#a78bfa'); ax.set_title('Elbow method'); ax.set_xlabel('Clusters (k)'); ax.set_ylabel('WCSS'); st.pyplot(fig); plt.close(fig)
with right:
    st.metric('Silhouette score',f'{sil:.3f}'); st.metric('Clusters',n_clusters); st.metric('Customers clustered',len(labels))

st.header('4. Cluster visualizations')
left,right=st.columns(2)
with left:
    pca=PCA(n_components=2); xp=pca.fit_transform(X_scaled); fig,ax=dark_fig()
    for i in sorted(set(labels)): ax.scatter(xp[labels==i,0],xp[labels==i,1],label=f'Cluster {i}',color=PALETTE[i%len(PALETTE)],alpha=.8)
    ax.set_title(f"PCA ({pca.explained_variance_ratio_.sum()*100:.1f}% variance)"); ax.legend(); st.pyplot(fig); plt.close(fig)
with right:
    fig,ax=dark_fig()
    for i in sorted(set(labels)): ax.scatter(X.iloc[labels==i,0],X.iloc[labels==i,1],label=f'Cluster {i}',color=PALETTE[i%len(PALETTE)],alpha=.8)
    centers=scaler.inverse_transform(model.cluster_centers_); ax.scatter(centers[:,0],centers[:,1],marker='X',s=200,c='white',label='Centroids'); ax.set_xlabel('Annual Income (k$)'); ax.set_ylabel('Spending Score'); ax.set_title('Customer segments'); ax.legend(); st.pyplot(fig); plt.close(fig)

st.header('5. Cluster statistics and recommendations')
result=processed.loc[X.index].copy(); result['Cluster']=labels
numeric=[c for c in ['Annual Income (k$)','Spending Score (1-100)','Age'] if c in result.columns]
summary=result.groupby('Cluster')[numeric].mean().round(2); st.dataframe(summary,use_container_width=True)
for cid,row in summary.iterrows():
    income=row['Annual Income (k$)']; spending=row['Spending Score (1-100)']
    if income>=60 and spending>=60: name,rec='High value','Premium membership and VIP rewards'
    elif income>=60: name,rec='Conservative','Exclusive discounts and targeted offers'
    elif spending>=60: name,rec='Enthusiastic','Loyalty points and cashback offers'
    else: name,rec='Budget','Basic deals and entry-level promotions'
    with st.expander(f'Cluster {cid}: {name}'):
        st.write(f'Average annual income: {income:.1f}k')
        st.write(f'Average spending score: {spending:.1f}')
        if 'Age' in row: st.write(f'Average age: {row["Age"]:.1f}')
        st.write('**Recommendation:** '+rec)
st.caption('Cluster labels are algorithmic groups; business profiles are heuristic suggestions based on average income and spending.')
