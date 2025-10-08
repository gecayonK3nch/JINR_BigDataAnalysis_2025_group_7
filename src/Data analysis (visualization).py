import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from matplotlib.patches import Rectangle
from scipy.stats import linregress

# Загрузка данных из файла
df = pd.read_csv('patents.csv')
print(df.info())

# Функция для извлечения авторов
def extract_authors(authors_str):
    if pd.isna(authors_str):
        return []
    authors_clean = re.sub(r'["\']', '', str(authors_str)).strip()
    return [author.strip() for author in authors_clean.split(',')]

# Анализ авторов
all_authors = []
for authors in df['authors']:
    all_authors.extend(extract_authors(authors))
author_counts = Counter(all_authors)
top_authors = author_counts.most_common(10)

# Количество авторов по патентам
df['num_authors'] = df['authors'].apply(lambda x: len(extract_authors(x)) if pd.notna(x) else 0)

# Парсинг дат подачи
df['date_clean'] = df['date'].astype(str).str.replace(',', '.')
df['date'] = pd.to_datetime(df['date_clean'], format='%d.%m.%Y', errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
yearly_patents = df.groupby('year').size()

# График 1: Патенты по годам (линейный)
plt.figure(figsize=(10, 5))
yearly_patents.plot(kind='line', marker='o')
plt.title('Количество патентов по годам')
plt.xlabel('Год')
plt.ylabel('Количество патентов')
plt.grid(True)
plt.savefig('patents_by_year.png')
plt.show()

# График 2: Топ-10 авторов (столбчатый)
fig, ax = plt.subplots(figsize=(10, 6))
authors_names = [name for name, _ in top_authors]
authors_counts = [count for _, count in top_authors]
ax.barh(authors_names, authors_counts)
ax.set_title('Топ-10 авторов по частоте упоминаний')
ax.set_xlabel('Количество упоминаний')
plt.tight_layout()
plt.savefig('top_authors.png')
plt.show()

# График 3: Распределение кол-ва авторов (гистограмма)
plt.figure(figsize=(8, 5))
df['num_authors'].hist(bins=range(1, 12), edgecolor='black')
plt.title('Распределение количества авторов по патентам')
plt.xlabel('Количество авторов')
plt.ylabel('Количество патентов')
plt.grid(True)
plt.savefig('authors_distribution.png')
plt.show()

# Извлечение года публикации
def extract_pub_year(pub_str):
    if pd.isna(pub_str):
        return None
    match = re.search(r'(\d{4})', str(pub_str))
    if match:
        return int(match.group(1))
    return None

df['pub_year'] = df['publication'].apply(extract_pub_year)

# Таблица патентов за 2025 год (по публикации)
patents_2025 = df[df['pub_year'] == 2025][['n_patent', 'title', 'authors', 'publication']]
print("Патенты за 2025 год:")
print(patents_2025.to_string(index=False))

# Тепловая карта 1: По годам и месяцам (подача)
pivot_year_month = df.pivot_table(values='n', index='year', columns='month', aggfunc='count', fill_value=0)
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_year_month, annot=True, cmap='YlOrRd', fmt='d')
plt.title('Тепловая карта патентов по годам и месяцам (подача)')
plt.xlabel('Месяц')
plt.ylabel('Год')
plt.savefig('heatmap_year_month.png')
plt.show()

# Тепловая карта 2: Топ-авторы vs Годы (эксплод авторов)
df_exploded = df.copy()
df_exploded['authors_list'] = df_exploded['authors'].apply(extract_authors)
df_exploded = df_exploded.explode('authors_list')
df_exploded['authors_clean'] = df_exploded['authors_list'].str.strip()
df_exploded = df_exploded.dropna(subset=['authors_clean'])
top_authors_list = df_exploded['authors_clean'].value_counts().head(10).index.tolist()
pivot_author_year = df_exploded[df_exploded['authors_clean'].isin(top_authors_list)].pivot_table(
    values='n', index='year', columns='authors_clean', aggfunc='count', fill_value=0
)
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_author_year, annot=True, cmap='Blues', fmt='d')
plt.title('Тепловая карта: Топ авторы vs Годы')
plt.xlabel('Год')
plt.ylabel('Авторы')
plt.savefig('heatmap_authors_year.png')
plt.show()

# Сегментация: По ключевым словам в title
keywords = {
    'Нейтроны': ['нейтрон', 'нейтронов'],
    'Детекторы': ['детектор', 'детекции'],
    'Ускорители': ['циклотрон', 'ускоритель', 'пучи'],
    'Сверхпроводимость': ['сверхпровод', 'сверхпроводящ'],
    'Биология': ['биолог', 'белок', 'ДНК', 'генет']
}

def segment_title(title):
    title_lower = str(title).lower()
    segments = []
    for seg, kws in keywords.items():
        if any(kw in title_lower for kw in kws):
            segments.append(seg)
    return ', '.join(segments) if segments else 'Другое'

df['segment'] = df['title'].apply(segment_title)
segment_counts = df['segment'].value_counts()
print("\nСегментация по темам:")
print(segment_counts.to_string())

# Тепловая карта 3: Сегменты vs Годы
pivot_segment_year = df.pivot_table(values='n', index='year', columns='segment', aggfunc='count', fill_value=0)
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_segment_year, annot=True, cmap='viridis', fmt='d')
plt.title('Тепловая карта: Сегменты по годам')
plt.xlabel('Сегмент')
plt.ylabel('Год')
plt.savefig('heatmap_segment_year.png')
plt.show()

# Новый раздел: Графики патентного ландшафта

# 1. Анализ лабораторий
lab_counts = df['main_lab'].value_counts().head(10)
plt.figure(figsize=(12, 8))  # Increased height to avoid tight_layout warning
lab_counts.plot(kind='bar')
plt.title('Топ-10 лабораторий по количеству патентов')
plt.xlabel('Лаборатория')
plt.ylabel('Количество патентов')
plt.xticks(rotation=90)
plt.tight_layout(pad=3.0)  # Added padding to handle the warning
plt.savefig('top_labs.png')
plt.show()

# Тепловая карта: Лаборатории vs Годы
df_lab_year = df.pivot_table(values='n', index='year', columns='main_lab', aggfunc='count', fill_value=0)
plt.figure(figsize=(14, 10))  # Increased size
sns.heatmap(df_lab_year, annot=True, cmap='coolwarm', fmt='d')
plt.title('Тепловая карта: Лаборатории по годам')
plt.xlabel('Лаборатория')
plt.ylabel('Год')
plt.xticks(rotation=90)
plt.tight_layout(pad=2.0)
plt.savefig('heatmap_lab_year.png')
plt.show()

# 2. Word Cloud для заголовков патентов (патентный ландшафт по ключевым словам)
text = ' '.join(df['title'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud: Ключевые слова в заголовках патентов')
plt.savefig('wordcloud_titles.png')
plt.show()

# 3. Bubble Chart: Год vs Сегмент vs Размер (количество)
segment_year_agg = df.groupby(['year', 'segment']).size().reset_index(name='count')
plt.figure(figsize=(12, 8))
for seg in segment_year_agg['segment'].unique():
    seg_data = segment_year_agg[segment_year_agg['segment'] == seg]
    plt.scatter(seg_data['year'], seg_data['count'], s=seg_data['count']*100, alpha=0.6, label=seg)
plt.title('Bubble Chart: Патентный ландшафт по годам и сегментам')
plt.xlabel('Год')
plt.ylabel('Количество')
plt.legend()
plt.grid(True)
plt.savefig('bubble_patent_landscape.png')
plt.show()

# 4. Сеть соавторов (упрощенная, топ-авторы)
G = nx.Graph()
top_authors_dict = dict(author_counts.most_common(20))
for idx, row in df.iterrows():
    authors = extract_authors(row['authors'])
    for i in range(len(authors)):
        if authors[i] in top_authors_dict:
            G.add_node(authors[i])
        for j in range(i+1, len(authors)):
            if authors[j] in top_authors_dict:
                G.add_edge(authors[i], authors[j])
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=8)
plt.title('Сеть соавторов (топ-20)')
plt.savefig('coauthor_network.png')
plt.show()

# 5. PCA для тематического ландшафта (TF-IDF на titles)
vectorizer = TfidfVectorizer(max_features=100, stop_words=None)
tfidf_matrix = vectorizer.fit_transform(df['title'].astype(str))
pca = PCA(n_components=2)
pca_result = pca.fit_transform(tfidf_matrix.toarray())
df_pca = pd.DataFrame({'PC1': pca_result[:, 0], 'PC2': pca_result[:, 1], 'year': df['year'], 'lab': df['main_lab']})
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['year'], s=50, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Год')
plt.title('PCA: Тематический патентный ландшафт (TF-IDF на заголовках)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.savefig('pca_patent_landscape.png')
plt.show()

# 6. Регрессия: Линейная тренд по годам
years = df['year'].dropna().unique()
counts = [df[df['year'] == y].shape[0] for y in sorted(years)]
slope, intercept, r_value, p_value, std_err = linregress(sorted(years), counts)
plt.figure(figsize=(10, 5))
plt.plot(sorted(years), counts, 'o', label='Данные')
trend_years = np.array(sorted(years))
trend = slope * trend_years + intercept
plt.plot(sorted(years), trend, 'r', label=f'Тренд (R²={r_value**2:.2f})')
plt.title('Линейная регрессия: Тренд патентов по годам')
plt.xlabel('Год')
plt.ylabel('Количество')
plt.legend()
plt.grid(True)
plt.savefig('regression_trend.png')
plt.show()

print("Анализ завершен. Все графики сохранены как PNG файлы.")