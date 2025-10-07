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
    import difflib
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
CODIFIER_KEYWORDS: dict[str, set[str]] = {
    "A": {
        "медицина", "медицинский", "здоровье", "фармацевтика", "фармацевтический",
        "лекарство", "лечение", "пациент", "диагностика", "сельский", "хозяйство",
        "аграрный", "agriculture", "food", "nutrition", "medicine", "health",
        "farm", "crop", "biotech", "medical", "клиника", "клинический",
        "здравоохранение", "ветеринар", "ветеринарный", "therapy", "therapeutic",
        "clinical", "biomedical", "diagnostic", "pharmacy", "treatment",
        "healthcare", "rehabilitation"
    },
    "B": {
        "транспорт", "логистика", "перевозка", "доставка", "упаковка", "конвейер",
        "манипулятор", "погрузчик", "handling", "shipping", "vehicle", "logistic",
        "transport", "экскаватор", "карьера", "траншея", "storage", "container",
        "склад", "складской", "warehouse", "груз", "грузовой", "freight",
        "транспортировка", "logistics", "supply chain", "material handling",
        "маршрут"
    },
    "C": {
        "химия", "химический", "реакция", "катализатор", "полимер", "сплав",
        "металл", "металлургия", "металлургический", "электролиз", "chemical",
        "chemistry", "alloy", "catalyst", "polymer", "metallurgy", "compound",
        "synthesis", "реагент", "смола", "органика", "органический", "композит",
        "композитный", "nanomaterial", "nanoparticle", "reaction",
        "material science", "неорганический"
    },
    "D": {
        "текстиль", "ткань", "нить", "волокно", "бумага", "целлюлоза", "картон",
        "пряжа", "weaving", "paper", "textile", "fiber", "yarn", "fabric",
        "пульпа", "ткацкий", "бумажный", "нетканый", "текстильный",
        "прядение", "картонный"
    },
    "E": {
        "строительство", "строительный", "здание", "сооружение", "бетон", "цемент",
        "архитектура", "фундамент", "кирпич", "building", "construction",
        "housing", "infrastructure", "bridge", "road", "градостроительство",
        "каркас", "монолит", "строить", "инфраструктурный", "строительно",
        "архитектурный", "civil engineering"
    },
    "F": {
        "механика", "инженерия", "машиностроение", "двигатель", "мотор", "насос",
        "турбина", "передача", "шестерня", "инструмент", "механизм", "engine",
        "machine", "mechanical", "gear", "pump", "tool", "станок", "робот",
        "инженерный", "привод", "робототехника", "машинный", "станочный",
        "actuator", "mechanism", "kinematics"
    },
    "G": {
        "физика", "оптика", "радиация", "лазер", "квантовый", "детектор",
        "измерение", "фотон", "атом", "ядерный", "спектроскопия", "quantum",
        "photon", "optics", "sensor", "nuclear", "physics", "магнитный",
        "излучение", "ускоритель", "ускорительный", "ускорение", "пучок",
        "частица", "частицы", "заряд", "заряженный", "beam", "accelerator",
        "particle", "ion", "ион", "циклотрон", "спектрометр", "спектрометрия",
        "радиометр", "лазерный", "детектирование", "измерительный"
    },
    "H": {
        "электричество", "электрический", "электроника", "электронный",
        "микросхема", "питание", "батарея", "аккумулятор", "генератор", "ток",
        "напряжение", "circuit", "power", "battery", "electrical", "electronic",
        "semiconductor", "microchip", "инвертор", "преобразователь",
        "силовой", "электропитание", "энергосистема", "power supply",
        "электротехника", "энергетика"
    },
}
CODIFIER_CODE_WEIGHTS: dict[str, float] = {
    "A": 1.1,
    "B": 1.0,
    "C": 1.05,
    "D": 1.0,
    "E": 1.05,
    "F": 1.0,
    "G": 0.55,
    "H": 1.0,
}
def _build_codifier_vectorizer() -> tuple[Any, Any, list[str]] | None:
    if TfidfVectorizer is None or cosine_similarity is None:
        return None
    codes = list(CODIFIER_KEYWORDS.keys())
    corpus = [" ".join(sorted(words)) for words in CODIFIER_KEYWORDS.values()]
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix, codes


CODIFIER_VECTOR_DATA = _build_codifier_vectorizer()


class _CodifierKeywordLookup:
    def __init__(
        self,
        keywords_map: dict[str, set[str]],
        vector_bundle: tuple[Any, Any, list[str]] | None,
    ):
        self._vector_bundle = vector_bundle
        self._cache: dict[str, tuple[tuple[str, float], ...]] = {}
        grouped: dict[str, list[tuple[str, float]]] = {}
        multiword: dict[int, set[str]] = {}
        for code, keywords in keywords_map.items():
            base_weight = CODIFIER_CODE_WEIGHTS.get(code, 1.0)
            for keyword in keywords:
                normalized = keyword.lower()
                grouped.setdefault(normalized, []).append((code, base_weight))
                if " " in normalized:
                    length = normalized.count(" ") + 1
                    multiword.setdefault(length, set()).add(normalized)
        self._direct: dict[str, tuple[tuple[str, float], ...]] = {
            token: tuple(values) for token, values in grouped.items()
        }
        self._known_tokens: tuple[str, ...] = tuple(self._direct.keys())
        self.multiword_by_len: dict[int, frozenset[str]] = {
            length: frozenset(values) for length, values in multiword.items()
        }
        self.multiword_tokens: frozenset[str] = frozenset(
            phrase for phrases in self.multiword_by_len.values() for phrase in phrases
        )

    def has_token(self, token: str) -> bool:
        return token.lower() in self._direct

    def lookup(self, token: str) -> tuple[tuple[str, float], ...]:
        if not token:
            return ()
        normalized = token.lower()
        direct = self._direct.get(normalized)
        if direct:
            return direct
        cached = self._cache.get(normalized)
        if cached is not None:
            return cached
        inferred = self._infer_from_vector(normalized)
        if inferred:
            self._cache[normalized] = inferred
            return inferred
        approx = self._approximate_lookup(normalized)
        self._cache[normalized] = approx
        return approx

    def _infer_from_vector(self, token: str) -> tuple[tuple[str, float], ...]:
        bundle = self._vector_bundle
        if bundle is None:
            return ()
        vectorizer, matrix, codes = bundle
        vector = vectorizer.transform([token])
        sims = cosine_similarity(vector, matrix).ravel()  # type: ignore
        weighted: list[tuple[str, float]] = []
        for code, score in zip(codes, sims):
            score_float = float(score)
            if score_float <= 0.0:
                continue
            base_weight = CODIFIER_CODE_WEIGHTS.get(code, 1.0)
            weighted.append((code, score_float * base_weight))
        if not weighted:
            return ()
        max_score = max(score for _, score in weighted)
        if max_score <= 0.0:
            return ()
        threshold = max_score * 0.72
        filtered = [
            (code, score)
            for code, score in weighted
            if score >= threshold or score == max_score
        ]
        filtered.sort(key=lambda item: (-item[1], item[0]))
        normalized = tuple((code, max(score, 0.2)) for code, score in filtered)
        return normalized

    def _approximate_lookup(self, token: str) -> tuple[tuple[str, float], ...]:
        if len(token) < 4 or not self._known_tokens:
            return ()
        candidates = difflib.get_close_matches(token, self._known_tokens, n=4, cutoff=0.78)
        if not candidates:
            return ()
        aggregated: dict[str, float] = {}
        for candidate in candidates:
            base_pairs = self._direct.get(candidate)
            if not base_pairs:
                continue
            similarity = difflib.SequenceMatcher(a=token, b=candidate).ratio()
            if similarity < 0.78:
                continue
            for code, base_weight in base_pairs:
                score = base_weight * similarity
                aggregated[code] = max(aggregated.get(code, 0.0), score)
        if not aggregated:
            return ()
        ranked = sorted(aggregated.items(), key=lambda item: (-item[1], item[0]))
        return tuple((code, max(weight, 0.2)) for code, weight in ranked[:3])


KEYWORD_TO_CODIFIERS = _CodifierKeywordLookup(CODIFIER_KEYWORDS, CODIFIER_VECTOR_DATA)


def classify_codifier(words: Iterable[str]) -> str | None:
    words_list = list(words)
    if not words_list:
        return None
    candidates: list[tuple[str, int, float]] = []
    for idx, word in enumerate(words_list):
        clean_len = len(word)
        if clean_len >= 10:
            multiplier = 1.2
        elif clean_len >= 7:
            multiplier = 1.1
        elif clean_len >= 5:
            multiplier = 1.05
        else:
            multiplier = 1.0
        candidates.append((word, idx, multiplier))
    multiword_by_len = KEYWORD_TO_CODIFIERS.multiword_by_len
    if multiword_by_len:
        max_len = max(multiword_by_len)
        for n in range(2, max_len + 1):
            phrases = multiword_by_len.get(n)
            if not phrases or len(words_list) < n:
                continue
            phrase_multiplier = 1.18 + 0.05 * min(n - 2, 2)
            for idx in range(len(words_list) - n + 1):
                phrase = " ".join(words_list[idx:idx + n])
                if phrase not in phrases:
                    continue
                candidates.append((phrase, idx, phrase_multiplier))
    counts: Counter[str] = Counter()
    first_occurrence: dict[str, int] = {}
    for term, idx, multiplier in candidates:
        matches = KEYWORD_TO_CODIFIERS.lookup(term)
        if not matches:
            continue
        direct_bonus = 1.05 if KEYWORD_TO_CODIFIERS.has_token(term) else 1.0
        for code, weight in matches:
            counts[code] += weight * multiplier * direct_bonus # type: ignore
            prev_idx = first_occurrence.get(code)
            if prev_idx is None or idx < prev_idx:
                first_occurrence[code] = idx
    if counts:
        code, _ = min(
            counts.items(),
            key=lambda item: (
                -item[1],
                first_occurrence.get(item[0], float("inf")),
                item[0],
            ),
        )
        return code
    if CODIFIER_VECTOR_DATA is None or not words_list:
        return None
    vectorizer, matrix, codes = CODIFIER_VECTOR_DATA
    text_vector = vectorizer.transform([" ".join(words_list)])
    best_code: str | None = None
    best_score = 0.0
    for code, similarity in zip(codes, cosine_similarity(text_vector, matrix).ravel()):  # type: ignore
        score = float(similarity) * CODIFIER_CODE_WEIGHTS.get(code, 1.0)
        if score > best_score:
            best_score = score
            best_code = code
    return best_code if best_score > 0.0 else None


def extract_patent_codifier(
    title: str,
    morph=None,
) -> str:
    words = tokenize(title, morph)
    code = classify_codifier(words)
    return code or ""


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
        similarities = cosine_similarity(text_vector, lab_matrix).ravel() #type: ignore
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
    codifier_column: str = "codifier",
) -> pd.DataFrame:
    # Robust CSV loading: try default, then try common encodings/delimiters.
    def _read_csv_flexible(path: Path) -> pd.DataFrame:
        # Try pandas default first
        try:
            return pd.read_csv(path)
        except pd.errors.ParserError:
            pass

        # Try common encodings and delimiters
        encodings = ("utf-8", "cp1251", "latin1")
        delimiters = [",", ";", "\t", "|"]
        import csv

        sample = None
        # Read a sample for sniffing
        try:
            with open(path, "rb") as fh:
                raw = fh.read(8192)
            # try to decode sample using common encodings
            for enc in encodings:
                try:
                    sample = raw.decode(enc)
                    detected_enc = enc
                    break
                except Exception:
                    continue
        except Exception:
            sample = None
            detected_enc = "utf-8"

        sep_candidates = list(delimiters)
        # If we have a text sample, try csv.Sniffer to detect delimiter
        if sample:
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=''.join(delimiters))
                if getattr(dialect, "delimiter", None):
                    sep_candidates.insert(0, dialect.delimiter)
            except Exception:
                # ignore sniff errors
                pass

        # Try combinations of encodings and separators with the python engine
        for enc in encodings:
            for sep in sep_candidates:
                try:
                    return pd.read_csv(path, sep=sep, encoding=enc, engine="python")
                except pd.errors.ParserError:
                    continue
                except Exception:
                    # other decoding/parsing errors — try next
                    continue

        # Last resort: let pandas try to infer with python engine and warn on bad lines
        try:
            return pd.read_csv(path, engine="python", sep=None, encoding="utf-8", on_bad_lines="warn")
        except Exception:
            # Final fallback: raise the parser error to the caller
            return pd.read_csv(path)

    df = _read_csv_flexible(csv_path)
    if title_column not in df.columns:
        raise ValueError(f"Column '{title_column}' missing in {csv_path}")
    morph = _load_morph_analyzer()
    df[codifier_column] = df[title_column].fillna("").apply(
        lambda txt: extract_patent_codifier(txt, morph=morph)
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
    parsed = parser.parse_args(args=args) # type: ignore
    out_path = parsed.output or parsed.input_csv
    annotate_patent_topics(
        csv_path=parsed.input_csv,
        output_path=out_path,
        title_column=parsed.title_column
        )


if __name__ == "__main__":
    main()