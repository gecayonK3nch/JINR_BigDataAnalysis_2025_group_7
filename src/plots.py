import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import io
from pathlib import Path

# Справочник разделов по кодировке (IPC)
CODIFIER_MAP = {
    'A': 'УДОВЛЕТВОРЕНИЕ ЖИЗНЕННЫХ ПОТРЕБНОСТЕЙ ЧЕЛОВЕКА',
    'B': 'РАЗЛИЧНЫЕ ТЕХНОЛОГИЧЕСКИЕ ПРОЦЕССЫ; ТРАНСПОРТИРОВАНИЕ',
    'C': 'ХИМИЯ; МЕТАЛЛУРГИЯ',
    'D': 'ТЕКСТИЛЬ; БУМАГА',
    'E': 'СТРОИТЕЛЬСТВО И ГОРНОЕ ДЕЛО',
    'F': 'МАШИНОСТРОЕНИЕ; ОСВЕЩЕНИЕ; ОТОПЛЕНИЕ; ОРУЖИЕ И БОЕПРИПАСЫ; ВЗРЫВНЫЕ РАБОТЫ',
    'G': 'ФИЗИКА',
    'H': 'ЭЛЕКТРИЧЕСТВО',
}

# Настройка страницы
st.set_page_config(
    page_title="Анализ патентов",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стиль для улучшенного отображения
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .patent-card {
        background-color: white;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Заголовок приложения
st.markdown('<h1 class="main-header">📊 Аналитическая панель патентов</h1>', unsafe_allow_html=True)

# Функция для загрузки данных
@st.cache_data
def load_data(uploaded_file):
    try:
        # Чтение CSV файла
        df = pd.read_csv(uploaded_file, encoding='utf-8')
        return df
        
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")
        return None

# Функция для обработки данных (без нормализации столбцов)
def process_data(df):
    # Очистка данных
    df = df.dropna(how='all')

    # Преобразование даты регистрации, если присутствует
    if 'registration_date' in df.columns:
        df['registration_date'] = pd.to_datetime(df['registration_date'], format='%d.%m.%Y', errors='coerce')

    # Приведение codifier к верхнему регистру
    if 'codifier' in df.columns:
        df['codifier'] = df['codifier'].astype(str).str.strip().str.upper()

    # Авторы как строка
    if 'authors' in df.columns:
        df['authors'] = df['authors'].astype(str)

    # Подразделение
    if 'subdivision' in df.columns:
        df['subdivision'] = df['subdivision'].fillna('Не указано').astype(str).str.strip()

    return df

# Боковая панель для загрузки файла
st.sidebar.title("📁 Загрузка данных")

data_source = st.sidebar.radio(
    "Источник данных",
    ["Загрузить файл", "Выбрать из ./data"],
    index=1
)

df = None

if data_source == "Загрузить файл":
    uploaded_file = st.sidebar.file_uploader(
        "Загрузите CSV файл с патентами",
        type=['csv'],
        help="Файл должен содержать колонки: certificate_number, registration_date, title, title_link, publication_ref, authors, subdivision, codifier"
    )
    if uploaded_file is not None:
        df_raw = load_data(uploaded_file)
        if df_raw is not None:
            df = process_data(df_raw)
else:
    data_dir = (Path(__file__).resolve().parent / '..' / 'data').resolve()
    available_csv_paths = sorted([p for p in data_dir.glob('*.csv')])
    if available_csv_paths:
        selected_filename = st.sidebar.selectbox("Файл из ./data:", [p.name for p in available_csv_paths])
        if selected_filename:
            selected_path = data_dir / selected_filename
            df_raw = load_data(str(selected_path))
            if df_raw is not None:
                df = process_data(df_raw)
    else:
        st.sidebar.info("В папке ./data нет CSV файлов")

# Если данные загружены, показываем аналитику
if 'df' in locals() and df is not None:
    
    # Фильтры в боковой панели
    st.sidebar.title("🔍 Фильтры")
    
    # Фильтр по лаборатории/подразделению
    if 'subdivision' in df.columns:
        subdivision_values = ['Все'] + sorted(df['subdivision'].unique().tolist())
        selected_subdivision = st.sidebar.selectbox(
            "Выберите лабораторию/подразделение:",
            subdivision_values
        )
    else:
        selected_subdivision = 'Все'
        st.sidebar.warning("Столбец 'subdivision' не найден в данных")

    # Фильтр по кодировке (IPC)
    selected_codifiers = []
    codifier_display_options = []
    available_codifiers = []
    if 'codifier' in df.columns:
        available_codifiers = sorted([c for c in df['codifier'].dropna().astype(str).str.upper().unique().tolist() if c in CODIFIER_MAP])
        codifier_display_options = [f"{c} - {CODIFIER_MAP.get(c, '')}" for c in available_codifiers]
        default_codifiers = codifier_display_options
        selected_codifier_display = st.sidebar.multiselect(
            "Выберите раздел(ы) по кодировке (IPC):",
            options=codifier_display_options,
            default=default_codifiers
        )
        selected_codifiers = [s.split(' - ')[0] for s in selected_codifier_display]
    
    # Фильтр по дате регистрации
    if 'registration_date' in df.columns:
        min_date = df['registration_date'].min()
        max_date = df['registration_date'].max()
    else:
        min_date = pd.NaT
        max_date = pd.NaT
    
    if not df.empty and pd.notna(min_date) and pd.notna(max_date):
        date_range = st.sidebar.date_input(
            "Диапазон дат подачи заявок",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        # Применение фильтров
        df_filtered = df.copy()

        # Фильтр по подразделению
        if selected_subdivision != 'Все' and 'subdivision' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['subdivision'] == selected_subdivision]
        # Фильтр по кодировке
        if 'codifier' in df_filtered.columns and available_codifiers:
            if selected_codifiers and len(selected_codifiers) < len(available_codifiers):
                df_filtered = df_filtered[df_filtered['codifier'].isin(selected_codifiers)]
        
        # Фильтр по дате регистрации
        if len(date_range) == 2 and 'registration_date' in df_filtered.columns:
            mask = (df_filtered['registration_date'] >= pd.to_datetime(date_range[0])) & (df_filtered['registration_date'] <= pd.to_datetime(date_range[1]))
            df_filtered = df_filtered[mask]
    else:
        df_filtered = df.copy()
        # Применение только фильтра по подразделению
        if selected_subdivision != 'Все' and 'subdivision' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['subdivision'] == selected_subdivision]
        # Фильтр по кодировке
        if 'codifier' in df_filtered.columns and available_codifiers:
            if selected_codifiers and len(selected_codifiers) < len(available_codifiers):
                df_filtered = df_filtered[df_filtered['codifier'].isin(selected_codifiers)]
        st.sidebar.warning("Некорректные даты в данных")
    
    # Показываем активные фильтры
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Активные фильтры")
    st.sidebar.write(f"**Лаборатория:** {selected_subdivision}")
    if 'date_range' in locals() and len(date_range) == 2:
        st.sidebar.write(f"**Период:** {date_range[0]} - {date_range[1]}")
    if 'codifier' in df.columns and available_codifiers:
        if not selected_codifiers or len(selected_codifiers) == len(available_codifiers):
            st.sidebar.write("**Кодировка (IPC):** Все")
        else:
            st.sidebar.write(f"**Кодировка (IPC):** {', '.join(selected_codifiers)}")
    st.sidebar.write(f"**Найдено записей:** {len(df_filtered)}")
    
    # Основная панель
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Обзор", "📊 Графики", "📋 Данные", "👥 Авторы", "🏢 Лаборатории"])

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Всего патентов", len(df_filtered))
        
        with col2:
            st.metric("Уникальных авторов", df_filtered['authors'].nunique())
        
        with col3:
            if 'registration_date' in df_filtered.columns and not df_filtered.empty and pd.notna(df_filtered['registration_date'].min()):
                st.metric("Первый патент", df_filtered['registration_date'].min().strftime('%d.%m.%Y'))
            else:
                st.metric("Первый патент", "Н/Д")

        with col4:
            if 'registration_date' in df_filtered.columns and not df_filtered.empty and pd.notna(df_filtered['registration_date'].max()):
                st.metric("Последний патент", df_filtered['registration_date'].max().strftime('%d.%m.%Y'))
            else:
                st.metric("Последний патент", "Н/Д")
        
        # Показываем информацию о лаборатории если фильтр применен
        if 'selected_subdivision' in locals() and selected_subdivision != 'Все':
            st.info(f"🔬 Отображены данные для лаборатории: **{selected_subdivision}**")
        
        # Карточки патентов с возможностью сворачивания
        st.subheader("🎯 Патенты")
        
        # Настройки отображения
        patents_to_show_initially = 3
        total_patents = len(df_filtered)
        
        if total_patents > 0:
            # Всегда показываем первые несколько патентов
            for i, (_, row) in enumerate(df_filtered.head(patents_to_show_initially).iterrows()):
                with st.container():
                    date_str = row['registration_date'].strftime('%d.%m.%Y') if ('registration_date' in row and pd.notna(row['registration_date'])) else "Н/Д"
                    main_lab_info = f"<p><strong>Лаборатория:</strong> {row['subdivision']}</p>" if 'subdivision' in row else ""
                    
                    st.markdown(f"""
                    <div class="patent-card">
                        <h4>{row['title']}</h4>
                        <p><strong>Номер:</strong> {row.get('certificate_number', '')}</p>
                        <p><strong>Дата регистрации:</strong> {date_str}</p>
                        <p><strong>Публикация/ссылка:</strong> {row.get('publication_ref', '')}</p>
                        {main_lab_info}
                        <p><strong>Авторы:</strong> {row['authors']}</p>
                        <p>{f'<a href="{row["title_link"]}" target="_blank">📄 Документ PDF</a>' if ('title_link' in row and str(row['title_link']).strip()) else ''}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Если патентов больше, чем изначально показываемых, добавляем аккордеон
            if total_patents > patents_to_show_initially:
                with st.expander(f"📂 Показать все патенты ({total_patents} всего)", expanded=False):
                    # Показываем оставшиеся патенты
                    for i, (_, row) in enumerate(df_filtered.iloc[patents_to_show_initially:].iterrows()):
                        with st.container():
                            date_str = row['registration_date'].strftime('%d.%m.%Y') if ('registration_date' in row and pd.notna(row['registration_date'])) else "Н/Д"
                            main_lab_info = f"<p><strong>Лаборатория:</strong> {row['subdivision']}</p>" if 'subdivision' in row else ""
                            
                            st.markdown(f"""
                            <div class="patent-card">
                                <h4>{row['title']}</h4>
                                <p><strong>Номер:</strong> {row.get('certificate_number', '')}</p>
                                <p><strong>Дата регистрации:</strong> {date_str}</p>
                                <p><strong>Публикация/ссылка:</strong> {row.get('publication_ref', '')}</p>
                                {main_lab_info}
                                <p><strong>Авторы:</strong> {row['authors']}</p>
                                <p>{f'<a href="{row["title_link"]}" target="_blank">📄 Документ PDF</a>' if ('title_link' in row and str(row['title_link']).strip()) else ''}</p>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.info("Нет патентов для отображения")
            
        with tab2:
            st.subheader("📊 Аналитические графики")
            
            if len(df_filtered) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Круговая диаграмма по месяцам
                    if 'registration_date' in df_filtered.columns and not df_filtered.empty and pd.notna(df_filtered['registration_date']).any():
                        df_filtered['month'] = df_filtered['registration_date'].dt.month
                        monthly_counts = df_filtered['month'].value_counts().sort_index()  # Сортировка по месяцам: 1, 2, ..., 12

                        if not monthly_counts.empty:
                            fig2 = px.pie(
                                values=monthly_counts.values,
                                names=monthly_counts.index,
                                title='Распределение патентов по месяцам',
                                category_orders={"names": list(monthly_counts.index)}  # Фиксируем порядок
                            )
                            fig2.update_layout(
                                legend=dict(
                                    traceorder="normal"  # Убедимся, что легенда идёт в том же порядке, что и данные
                                )
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                
                with col2:
                    # Гистограмма по датам
                    if 'registration_date' in df_filtered.columns and not df_filtered.empty and pd.notna(df_filtered['registration_date']).any():
                        fig3 = px.histogram(df_filtered, x='registration_date', title='Распределение патентов по датам регистрации',
                                        labels={'registration_date': 'Дата регистрации', 'count': 'Количество патентов'})
                        st.plotly_chart(fig3, use_container_width=True)
                    
                    # Heatmap активности
                    if 'registration_date' in df_filtered.columns and not df_filtered.empty and pd.notna(df_filtered['registration_date']).any():
                        df_filtered['year'] = df_filtered['registration_date'].dt.year
                        df_filtered['month_num'] = df_filtered['registration_date'].dt.month
                        heatmap_data = df_filtered.groupby(['year', 'month_num']).size().unstack(fill_value=0)
                        if not heatmap_data.empty:
                            fig4 = px.imshow(heatmap_data, title='Тепловая карта активности по месяцам и годам',
                                            labels=dict(x="Месяц", y="Год", color="Количество патентов"))
                            st.plotly_chart(fig4, use_container_width=True)
            else:
                st.warning("Нет данных для построения графиков")

    with tab3:
        st.subheader("📋 Полные данные")
        
        # Поиск
        search_term = st.text_input("🔍 Поиск по названию или авторам:")
        
        if search_term:
            filtered_df = df_filtered[df_filtered['title'].str.contains(search_term, case=False, na=False) | 
                            df_filtered['authors'].str.contains(search_term, case=False, na=False)]
        else:
            filtered_df = df_filtered
        
        # Отображение таблицы
        if not filtered_df.empty:
            display_df = filtered_df.copy()
            if 'registration_date' in display_df.columns:
                display_df['registration_date'] = display_df['registration_date'].dt.strftime('%d.%m.%Y')
            
            # Колонки для отображения
            columns_to_display = []
            for col in ['certificate_number', 'title', 'registration_date', 'publication_ref', 'authors', 'subdivision', 'codifier']:
                if col in display_df.columns:
                    columns_to_display.append(col)
            
            st.dataframe(
                display_df[columns_to_display],
                use_container_width=True,
                height=400
            )
            
            # Кнопка скачивания
            csv = filtered_df.to_csv(index=False, encoding='utf-8')
            st.download_button(
                label="📥 Скачать данные как CSV",
                data=csv,
                file_name="patents_data.csv",
                mime="text/csv",
            )
        else:
            st.info("Нет данных для отображения")

    with tab4:
        st.subheader("👥 Анализ авторов")
        
        if not df_filtered.empty:
            # Анализ авторов
            all_authors = []
            for authors_str in df_filtered['authors']:
                if pd.notna(authors_str):
                    authors = [author.strip() for author in str(authors_str).split(',')]
                    all_authors.extend(authors)
            
            if all_authors:
                author_counts = Counter(all_authors)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Топ авторов
                    top_authors = pd.DataFrame(author_counts.most_common(10), columns=['Автор', 'Количество патентов'])
                    fig5 = px.bar(top_authors, x='Количество патентов', y='Автор', 
                                title='Топ-10 самых активных авторов',
                                orientation='h')
                    st.plotly_chart(fig5, use_container_width=True)
                
                with col2:
                    # Распределение по количеству авторов
                    df_filtered['author_count'] = df_filtered['authors'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
                    author_count_dist = df_filtered['author_count'].value_counts().sort_index()
                    fig6 = px.pie(values=author_count_dist.values, names=author_count_dist.index,
                                title='Распределение по количеству авторов в патенте')
                    st.plotly_chart(fig6, use_container_width=True)
                
                # Детальная информация по авторам
                st.subheader("Детальная статистика авторов")
                selected_author = st.selectbox("Выберите автора для детального анализа:", sorted(author_counts.keys()))
                
                if selected_author:
                    author_patents = df_filtered[df_filtered['authors'].str.contains(selected_author, na=False)]
                    st.write(f"**Патенты автора {selected_author}:**")
                    for _, patent in author_patents.iterrows():
                        date_str = patent['registration_date'].strftime('%d.%m.%Y') if ('registration_date' in patent and pd.notna(patent['registration_date'])) else "Н/Д"
                        lab_info = f" ({patent['subdivision']})" if 'subdivision' in patent else ""
                        st.write(f"- {patent['title']} ({date_str}){lab_info}")
            else:
                st.info("Нет данных об авторах")
        else:
            st.warning("Нет данных для анализа авторов")

    with tab5:
        st.subheader("🏢 Анализ по лабораториям")
        
        if 'subdivision' in df.columns:
            # Статистика по лабораториям
            lab_stats = df['subdivision'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Круговая диаграмма по лабораториям
                fig7 = px.pie(values=lab_stats.values, names=lab_stats.index,
                            title='Распределение патентов по лабораториям')
                st.plotly_chart(fig7, use_container_width=True)
            
            with col2:
                # Столбчатая диаграмма по лабораториям
                fig8 = px.bar(x=lab_stats.index, y=lab_stats.values,
                            title='Количество патентов по лабораториям',
                            labels={'x': 'Лаборатория', 'y': 'Количество патентов'})
                fig8.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig8, use_container_width=True)
            
            # Детальная статистика по лабораториям
            st.subheader("Детальная статистика по лабораториям")
            for lab in lab_stats.index:
                lab_data = df[df['subdivision'] == lab]
                with st.expander(f"🔬 {lab} ({len(lab_data)} патентов)"):
                    if 'registration_date' in lab_data.columns:
                        min_d = lab_data['registration_date'].min()
                        max_d = lab_data['registration_date'].max()
                        min_s = min_d.strftime('%d.%m.%Y') if pd.notna(min_d) else 'Н/Д'
                        max_s = max_d.strftime('%d.%m.%Y') if pd.notna(max_d) else 'Н/Д'
                        st.write(f"**Период активности:** {min_s} - {max_s}")
                    st.write(f"**Уникальных авторов:** {lab_data['authors'].nunique()}")
                    
                    # Топ авторов в лаборатории
                    lab_authors = []
                    for authors_str in lab_data['authors']:
                        if pd.notna(authors_str):
                            authors = [author.strip() for author in str(authors_str).split(',')]
                            lab_authors.extend(authors)
                    
                    if lab_authors:
                        top_lab_authors = Counter(lab_authors).most_common(5)
                        st.write("**Топ авторов:**")
                        for author, count in top_lab_authors:
                            st.write(f"- {author} ({count} патентов)")
        else:
            st.warning("Столбец 'subdivision' не найден в данных")

    # Ландшафт в нижней части
    st.markdown("---")
    st.subheader("🌐 Ландшафт патентной активности")

    if not df_filtered.empty and 'authors' in df_filtered.columns:
        # Создание ландшафта с использованием облака точек
        fig_landscape = go.Figure()

        # Анализ авторов для ландшафта
        all_authors_landscape = []
        for authors_str in df_filtered['authors']:
            if pd.notna(authors_str):
                authors = [author.strip() for author in str(authors_str).split(',')]
                all_authors_landscape.extend(authors)
        
        if all_authors_landscape:
            author_counts_landscape = Counter(all_authors_landscape)
            
            # Создание искусственного ландшафта на основе данных
            for i, (author, count) in enumerate(author_counts_landscape.most_common(5)):
                author_patents = df_filtered[df_filtered['authors'].str.contains(author, na=False)]

                fig_landscape.add_trace(go.Scatter(
                    x=author_patents['registration_date'] if 'registration_date' in author_patents.columns else None,
                    y=[i] * len(author_patents),
                    mode='markers',
                    name=author,
                    marker=dict(size=15, opacity=0.7),
                    text=author_patents['title'],
                    hovertemplate='<b>%{text}</b><br>Автор: ' + author + '<br>Дата: %{x}<extra></extra>'
                ))

            fig_landscape.update_layout(
                title='Ландшафт патентной активности по топ-5 авторам',
                xaxis_title='Дата регистрации',
                yaxis_title='Авторы',
                showlegend=True,
                height=400
            )

            st.plotly_chart(fig_landscape, use_container_width=True)
        else:
            st.info("Нет данных для построения ландшафта")
    else:
        st.info("Загрузите данные для отображения ландшафта")

else:
    st.info("👈 Пожалуйста, загрузите CSV файл с данными патентов в боковой панели")

# Информация о приложении
st.sidebar.markdown("---")
st.sidebar.info("""
**Инструкция по использованию:**
1. Загрузите CSV файл через боковую панель
2. Файл должен содержать колонки: certificate_number, registration_date, title, title_link, publication_ref, authors, subdivision, codifier
3. Используйте фильтры для настройки отображения
4. Переключайтесь между вкладками для разного типа анализа
""")