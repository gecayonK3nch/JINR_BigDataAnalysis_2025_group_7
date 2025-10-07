from __future__ import annotations
import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("sklearn not found, TF-IDF based classification will be disabled.")
    TfidfVectorizer = None  # type: ignore[assignment]
    cosine_similarity = None  # type: ignore[assignment]

RUSSIAN_STOPWORDS: set[str] = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
    "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к",
    "у", "же", "вы", "за", "бы", "по", "ее", "мне", "есть", "нет",
    "для", "мы", "тебя", "их", "вы", "же", "из", "уже", "при", "без",
    "над", "об", "до", "ни", "под", "через", "или", "если", "ли",
    "быть", "будет", "также", "где", "когда", "чтобы", "кем", "чем",
    "этот", "эта", "эти", "того", "тому", "та", "тот", "тем", "там",
    "они", "оно"
}

LAB_KEYWORDS: dict[str, set[str]] = {
    "Лаборатория физики высоких энергий им. В.И. Векслера и А.М. Балдина": {
        "высокоэнергетический", "высокая", "энергия", "high", "energy", "physics",
        "qcd", "адрон", "hadron", "коллайдер", "beam", "протон", "quark", "lepton",
        "кварк", "лептон", "нейтрино", "детектор", "tracker", "калориметр", "trigger",
        "beamline", "synchrotron", "accelerator", "luminosity", "cross", "section",
        "монте", "carlo", "симуляция", "частица", "collision", "cms", "lhc"
    },
    "Лаборатория ядерных проблем им. В.П. Джелепова": {
        "ядерный", "проблема", "нейтрино", "нейтрон", "радиация", "радиационный",
        "облучение", "дозиметрия", "камера", "большой", "volume", "детектор",
        "нейтронография", "флуоресценция", "силовая", "spectrometer", "spe", "tritium",
        "изотоп", "радионуклид", "гамма", "деконтоминация", "рентген", "нейтронография",
        "генератор", "модуль", "radiation", "monitoring", "shielding", "донный"
    },
    "Лаборатория теоретической физики им. Н.Н. Боголюбова": {
        "теоретический", "физика", "модель", "квантовый", "квантовая", "поле",
        "field", "theory", "симметрия", "string", "струна", "формализм", "уравнение",
        "шредингер", "лиагруппа", "механика", "калибровочный", "фазовый",
        "флуктуация", "lattice", "решетка", "космология", "инфляция", "гамильтонов",
        "лагранжиан", "квантование", "perturbation", "integrable", "аналитический",
        "кварк", "spinor", "брейн"
    },
    "Лаборатория нейтронной физики им. И.М. Франка": {
        "нейтрон", "нейтронный", "дифракция", "рассеяние", "scattering", "beam",
        "реактор", "pulst", "пучок", "cold", "thermal", "moderator", "instrument",
        "time", "flight", "tof", "фрактография", "reflectometry", "динамика",
        "структура", "lans", "birefringence", "lnp", "spectrometer", "дифрактометр",
        "powder", "инструмент", "нейтронография", "фотон"
    },
    "Лаборатория ядерных реакций им. Г.Н. Флерова": {
        "ядерный", "реакция", "реактор", "нейтрон", "изотоп", "радиоактивный",
        "радиация", "облучение", "пучок", "beam", "коллайдер", "fusion", "fission",
        "деление", "синтез", "heavy", "ion", "ион", "ускоритель", "accelerator",
        "target", "магнитный", "спектрометр", "детектор", "секция", "nuclear",
        "cross", "section", "плазма", "радионуклид", "thermonuclear", "гамма",
        "неутронный", "реактив", "superheavy"
    },
    "Лаборатория информационных технологий им. М.Г. Мещерякова": {
        "информация", "информационный", "информатика", "технология", "вычислительный",
        "данные", "аналитика", "анализ", "алгоритм", "машинный", "обучение", "deep",
        "learning", "нейросеть", "нейронный", "программирование", "software",
        "hardware", "data", "dataset", "цифровой", "цифровизация", "кластер",
        "кластерный", "cloud", "облако", "distributed", "распределенный",
        "параллельный", "кибербезопасность", "cybersecurity", "криптография",
        "искусственный", "intelligence", "ai", "ml", "computer", "vision",
        "распознавание", "speech", "nlp", "database", "база", "робот",
        "автоматизация", "информационно"
    },
    "Лаборатория радиационной биологии": {
        "радиация", "биология", "биологический", "клетка", "клеточный", "dna",
        "rna", "ген", "генетика", "мутация", "репарация", "радиобиология",
        "облучение", "доза", "дозиметрия", "шишка", "организм", "ткань", "живой",
        "молекула", "microbeam", "радионуклид", "sievert", "radiation", "health",
        "risk", "иммунный", "стресс", "экспозиция", "биомаркер", "apoptosis"
    },
    "Учебно-научный центр ОИЯИ": {
        "образование", "учебный", "центр", "студент", "курс", "школа", "лекция",
        "семинар", "практикум", "internship", "стажировка", "программа", "магистратура",
        "бакалавриат", "аспирантура", "training", "education", "summer", "school",
        "workshop", "конференция", "подготовка", "академия", "педагогика", "университет",
        "teacher", "mentoring", "competence", "skills", "проект", "olympiad"
    },
}

KEYWORD_TO_LABS: dict[str, set[str]] = {}
for lab, keywords in LAB_KEYWORDS.items():
    for keyword in keywords:
        KEYWORD_TO_LABS.setdefault(keyword, set()).add(lab)


def _build_lab_vectorizer() -> tuple[Any, Any, list[str]] | None:
    if TfidfVectorizer is None or cosine_similarity is None:
        return None
    lab_names = list(LAB_KEYWORDS.keys())
    corpus = [" ".join(sorted(words)) for words in LAB_KEYWORDS.values()]
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
    lab_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, lab_matrix, lab_names


LAB_VECTOR_DATA = _build_lab_vectorizer()


def _load_morph_analyzer():
    try:
        import pymorphy2  # type: ignore
    except ImportError:
        return None
    return pymorphy2.MorphAnalyzer()  # type: ignore


def tokenize(text: str, morph) -> list[str]:
    tokens = re.findall(r"[а-яёa-z]+", text.lower())
    lemmas: list[str] = []
    for token in tokens:
        if morph is not None:
            lemmas.append(morph.parse(token)[0].normal_form)
        else:
            lemmas.append(token)
    return [
        lemma
        for lemma in lemmas
        if lemma not in RUSSIAN_STOPWORDS and len(lemma) > 2
    ]


def classify_labs(
    words: Iterable[str],
    top_k: int,
    vector_data: tuple[Any, Any, list[str]] | None = None,
) -> list[str]:
    words_list = list(words)
    lab_counts: Counter[str] = Counter()
    first_occurrence: dict[str, int] = {}
    for idx, word in enumerate(words_list):
        for lab in KEYWORD_TO_LABS.get(word, ()):
            lab_counts[lab] += 1
            first_occurrence.setdefault(lab, idx)

    score_map: dict[str, float] = {lab: float(count) for lab, count in lab_counts.items()}
    vector_bundle = vector_data if vector_data is not None else LAB_VECTOR_DATA

    if vector_bundle is not None and words_list:
        vectorizer, lab_matrix, lab_names = vector_bundle
        text = " ".join(words_list)
        text_vector = vectorizer.transform([text])
        similarities = cosine_similarity(text_vector, lab_matrix).ravel()
        for lab, similarity in zip(lab_names, similarities):
            if similarity <= 0:
                continue
            score_map[lab] = score_map.get(lab, 0.0) + float(similarity)

    if not score_map:
        return []

    sorted_labs = sorted(
        score_map.items(),
        key=lambda item: (-item[1], first_occurrence.get(item[0], float("inf"))),
    )
    return [lab for lab, _ in sorted_labs[:top_k]]


def extract_main_topics(
    title: str,
    top_k: int = 3,
    morph=None,
    vector_data: tuple[Any, Any, list[str]] | None = None,
) -> str:
    words = tokenize(title, morph)
    labs = classify_labs(words, top_k, vector_data=vector_data)
    return ", ".join(labs)


def annotate_patent_topics(
    csv_path: Path,
    output_path: Path | None = None,
    title_column: str = "title",
    topic_column: str = "main_lab",
    top_k: int = 3,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if title_column not in df.columns:
        raise ValueError(f"Column '{title_column}' missing in {csv_path}")
    morph = _load_morph_analyzer()
    df[topic_column] = df[title_column].fillna("").apply(
        lambda txt: extract_main_topics(txt, top_k=top_k, morph=morph, vector_data=LAB_VECTOR_DATA)
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Добавляет колонку с лабораторией патента по его названию."
    )
    parser.add_argument("input_csv", type=Path, help="Путь к исходному CSV.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Путь для сохранения расширенного CSV (по умолчанию перезаписывает входной).",
    )
    parser.add_argument(
        "--title-column",
        default="title",
        help="Имя колонки с названиями патентов.",
    )
    parser.add_argument(
        "--topic-column",
        default="main_lab",
        help="Имя новой колонки для лаборатории.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Количество лабораторий в результирующей записи.",
    )
    return parser


def main(args: Iterable[str] | None = None) -> None:
    parser = _build_arg_parser()
    parsed = parser.parse_args(args=args)
    out_path = parsed.output or parsed.input_csv
    annotate_patent_topics(
        csv_path=parsed.input_csv,
        output_path=out_path,
        title_column=parsed.title_column,
        topic_column=parsed.topic_column,
        top_k=parsed.top_k,
    )


if __name__ == "__main__":
    main()