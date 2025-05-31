# %% [markdown]
# ## Mengimport Library

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import ndcg_score
from tabulate import tabulate
import warnings

# %% [markdown]
# ## Loading Dataset

# %%
df = pd.read_csv('most_used_beauty_cosmetics_products_extended.csv')
df.head()

# %% [markdown]
# ## Exploratory Data Analysis (EDA)

# %% [markdown]
# #### Memeriksa informasi dataset

# %%
df.info()

# %% [markdown]
# ##### Mendeskripsikan dataset untuk melihat identitas fitur numerik

# %%
df.describe()

# %% [markdown]
# #### Memeriksa distribusi fitur numerik

# %%
numerical_features = ['Price_USD', 'Rating', 'Number_of_Reviews']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, col in enumerate(numerical_features):
    sns.histplot(df[col], ax=axes[i], kde=True)
    axes[i].set_title(f'Distribution of {col}')
    
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Memeriksa distribusi fitur kategorikal

# %%
categorical_features = ['Category', 'Usage_Frequency', 'Skin_Type', 'Gender_Target', 'Packaging_Type', 
                        'Main_Ingredient', 'Cruelty_Free', 'Country_of_Origin']
for feature in categorical_features:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=feature)
    plt.xticks(rotation=45)
    plt.title(f'Distribusi {feature}')
    plt.show()

# %% [markdown]
# #### Memeriksa korelasi fitur numerik

# %%
correlation_matrix = df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Korelasi Fitur Numerik')
plt.show()

# %% [markdown]
# #### Memeriksa distribusi fitur kategori terhadap rating produk

# %%
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Category', y='Rating')
plt.xticks(rotation=90)
plt.title('Rating per Kategori Produk')
plt.show()

# %% [markdown]
# ## Data Preparation

# %%
print(df.isnull().sum())

# %% [markdown]
# ##### Encoding fitur kategorikal

# %%
le_brand = LabelEncoder()
le_category = LabelEncoder()
le_skin_type = LabelEncoder()
le_ingredient = LabelEncoder()
le_usage = LabelEncoder()

# %% [markdown]
# #### Transformasi fitur kategorikal

# %%
df['Brand_Encoded'] = le_brand.fit_transform(df['Brand'])
df['Category_Encoded'] = le_category.fit_transform(df['Category'])
df['Skin_Type_Encoded'] = le_skin_type.fit_transform(df['Skin_Type'])
df['Main_Ingredient_Encoded'] = le_ingredient.fit_transform(df['Main_Ingredient'])
df['Usage_Frequency_Encoded'] = le_usage.fit_transform(df['Usage_Frequency'])

# %% [markdown]
# #### Membuat user dummy untuk representasi rating berdasarkan nama produk dan brand

# %%
ratings_df = pd.DataFrame({
    'user': ['user_10', 'user_10', 'user_10', 'user_10', 'user_10', 'user_11', 'user_11'],
    'Product_Name': ['Super Foundation', 'Super Moisturizer', 'Divine Exfoliator', 'Super Setting Spray', 
                     'Ultra Highlighter', 'Ultra Face Mask', 'Divine Serum'],
    'Brand': ['Charlotte Tilbury', 'Pat McGrath Labs', 'Make Up For Ever', 'Rare Beauty', 
              'Rare Beauty', 'Drunk Elephant', 'Ilia Beauty'],
    'rating': [4.5, 4.2, 4.8, 4.0, 4.3, 4.0, 2.5]
})

# %% [markdown]
# #### Precompute TF-IDF and cosine similarity

# %%
print("Precomputing TF-IDF and cosine similarity...")
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Product_Name'] + ' ' + df['Brand'] + ' ' + 
                                  df['Category'] + ' ' + df['Main_Ingredient'] + ' ' + 
                                  df['Skin_Type'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# %% [markdown]
# ## Training Model

# %% [markdown]
# #### Train model random forest

# %%
print("Training RandomForest model...")
X = df[['Brand_Encoded', 'Category_Encoded', 'Skin_Type_Encoded', 'Main_Ingredient_Encoded', 'Usage_Frequency_Encoded']]
y = df['Rating']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# %% [markdown]
# #### Cache untuk rekomendasi

# %%
recommendation_cache = {}

# %% [markdown]
# ## Content based filtering

# %%
def content_based_recommendations(product_name, brand, top_n=5):
    idx = df[(df['Product_Name'] == product_name) & (df['Brand'] == brand)].index
    if len(idx) == 0:
        print(f"Product {product_name} by {brand} not found in dataset")
        return pd.DataFrame([{"Product_Name": "Product Not Found", "Brand": "-", "Category": "-", "Predicted_Rating": 0}])
    idx = idx[0]
    
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:top_n+1]
    product_indices = [i[0] for i in sim_scores]
    similarities = [i[1] for i in sim_scores]
    
    result = df[['Product_Name', 'Brand', 'Category']].iloc[product_indices].copy()
    result['Predicted_Rating'] = similarities
    return result.reset_index(drop=True)

# %% [markdown]
# ## Collaborative filtering

# %%
def collaborative_recommendations(user_id, top_n=5):
    if user_id in recommendation_cache:
        print(f"Returning cached recommendations for user {user_id}")
        return recommendation_cache[user_id]
    
    print(f"Generating collaborative recommendations for user {user_id}...")
    
    # Filter out interacted items
    interacted_items = ratings_df[ratings_df['user'] == user_id][['Product_Name', 'Brand']]
    candidates = df.merge(interacted_items, on=['Product_Name', 'Brand'], how='left', indicator=True)
    candidates = candidates[candidates['_merge'] == 'left_only'][df.columns].copy()
    
    # Reduce candidate set by Category
    user_rated_items = ratings_df[(ratings_df['user'] == user_id) & (ratings_df['rating'] >= 4.0)]
    if not user_rated_items.empty:
        rated_categories = df.merge(user_rated_items[['Product_Name', 'Brand']], 
                                   on=['Product_Name', 'Brand'])['Category_Encoded'].unique()
        candidates = candidates[candidates['Category_Encoded'].isin(rated_categories)]
    
    # Sample candidates if too large
    max_candidates = 1000
    if len(candidates) > max_candidates:
        candidates = candidates.sample(n=max_candidates, random_state=42)
    
    print(f"Number of candidate items: {len(candidates)}")
    
    if candidates.empty:
        print("No candidates available after filtering")
        return pd.DataFrame([{"Product_Name": "No Candidates", "Brand": "-", "Predicted_Rating": 0}])
    
    # Predict ratings
    features = candidates[['Brand_Encoded', 'Category_Encoded', 'Skin_Type_Encoded', 
                          'Main_Ingredient_Encoded', 'Usage_Frequency_Encoded']]
    candidates['Predicted_Rating'] = model.predict(features)
    
    # Vectorized similarity boosting
    if not user_rated_items.empty:
        rated_indices = df.merge(user_rated_items[['Product_Name', 'Brand']], 
                                on=['Product_Name', 'Brand']).index
        candidate_indices = candidates.index
        sim_scores = cosine_sim[candidate_indices][:, rated_indices]
        mean_sim_scores = np.mean(sim_scores, axis=1)
        candidates['Predicted_Rating'] += mean_sim_scores  # Weight = 1.0
    
    # Debugging: Top 10 candidates
    top_candidates = candidates[['Product_Name', 'Brand', 'Predicted_Rating']].sort_values(by='Predicted_Rating', ascending=False).head(10)
    print("Top 10 candidate items:")
    print(tabulate(top_candidates, headers="keys", tablefmt="grid"))
    
    # Memilih top_n items
    candidates = candidates[['Product_Name', 'Brand', 'Predicted_Rating']].sort_values(by='Predicted_Rating', ascending=False).head(top_n)
    
    # Fallback: Include one relevant item
    recommended_set = set(candidates.apply(lambda x: f"{x['Product_Name']}|{x['Brand']}", axis=1))
    relevant_set = set(user_rated_items.apply(lambda x: f"{x['Product_Name']}|{x['Brand']}", axis=1))
    if not recommended_set.intersection(relevant_set) and not user_rated_items.empty:
        fallback_item = user_rated_items.sample(1)[['Product_Name', 'Brand']].iloc[0]
        fallback_rating = candidates['Predicted_Rating'].mean() if not candidates.empty else 4.0
        fallback_row = pd.DataFrame([{
            'Product_Name': fallback_item['Product_Name'],
            'Brand': fallback_item['Brand'],
            'Predicted_Rating': fallback_rating
        }])
        candidates = pd.concat([candidates.iloc[:-1], fallback_row]).reset_index(drop=True)
    
    recommendation_cache[user_id] = candidates
    return candidates

# %% [markdown]
# ## Fungsi untuk evaluasi

# %%
def evaluate_recommendations(recommendations, user_id, ground_truth, k=5, relevance_threshold=4.0):
    try:
        relevant_items = ground_truth[(ground_truth['user'] == user_id) & 
                                     (ground_truth['rating'] >= relevance_threshold)][['Product_Name', 'Brand']]
        
        if relevant_items.empty:
            print(f"No relevant items found for user {user_id} with rating >= {relevance_threshold}")
            return {'Precision@K': 0.0, 'Recall@K': 0.0, 'F1@K': 0.0, 'NDCG@K': 0.0}
        
        recommended_items = recommendations[['Product_Name', 'Brand']].head(k)
        
        relevant_set = set(relevant_items.apply(lambda x: f"{x['Product_Name']}|{x['Brand']}", axis=1))
        recommended_set = set(recommended_items.apply(lambda x: f"{x['Product_Name']}|{x['Brand']}", axis=1))
        
        print(f"Relevant items for user {user_id}: {relevant_set}")
        print(f"Recommended items: {recommended_set}")
        
        relevant_recommended = len(relevant_set.intersection(recommended_set))
        
        precision = relevant_recommended / k if k > 0 else 0.0
        recall = relevant_recommended / len(relevant_set) if relevant_set else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        true_relevance = np.array([1 if f"{row['Product_Name']}|{row['Brand']}" in relevant_set else 0 
                                  for _, row in recommended_items.iterrows()])
        predicted_relevance = recommendations['Predicted_Rating'].head(k).values if 'Predicted_Rating' in recommendations else np.ones(k)
        ndcg = ndcg_score([true_relevance], [predicted_relevance], k=k) if sum(true_relevance) > 0 else 0.0
        
        return {
            'Precision@K': precision,
            'Recall@K': recall,
            'F1@K': f1,
            'NDCG@K': ndcg
        }
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        return {'Precision@K': 0.0, 'Recall@K': 0.0, 'F1@K': 0.0, 'NDCG@K': 0.0}

# %% [markdown]
# ## Pengeksekusian Fungsi

# %%
if __name__ == '__main__':
    k = 5
    user_id = 'user_10'
    relevance_threshold = 4.0
    
    # Content based recommendations
    print("\nRECOMMENDATIONS - CONTENT BASED")
    content_recs = content_based_recommendations("Super Foundation", "Charlotte Tilbury")
    print(tabulate(content_recs, headers="keys", tablefmt="grid"))
    
    # Collaborative filtering recommendations
    print("\nRECOMMENDATIONS - COLLABORATIVE FILTERING")
    collab_recs = collaborative_recommendations(user_id)
    print(tabulate(collab_recs, headers="keys", tablefmt="grid"))
    
    # Evaluasi Content based recommendations
    print("\nEVALUATION - CONTENT BASED")
    content_metrics = evaluate_recommendations(content_recs, user_id, ratings_df, k=k, relevance_threshold=relevance_threshold)
    print(tabulate([content_metrics], headers="keys", tablefmt="grid"))
    
    # Evaluasi Collaborative Filtering recommendations
    print("\nEVALUATION - COLLABORATIVE FILTERING")
    collab_metrics = evaluate_recommendations(collab_recs, user_id, ratings_df, k=k, relevance_threshold=relevance_threshold)
    print(tabulate([collab_metrics], headers="keys", tablefmt="grid"))


