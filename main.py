# ============================================
# Authorship Verification: Tacitus vs Dialogus
# O1 + O2 with unified word + char n-grams
# Optional segmentation + visualizations
# ============================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances


# ========= CONFIGURACIÓN GENERAL =========

USE_SEGMENTATION = True
SEGMENT_SIZE = 1174
O2_ITERATIONS = 500
RANDOM_SEED = 42
WORD_NGRAM_RANGE = (1, 1)
CHAR_NGRAM_RANGE = (3, 3)

# Solo palabras
# USE_WORDS = True
# USE_CHARS = False

# Solo caracteres 3-grams
# USE_WORDS = False
# USE_CHARS = True
# CHAR_NGRAM_RANGE = (3, 4)

# Solo caracteres 2–4
USE_WORDS = False
USE_CHARS = True
CHAR_NGRAM_RANGE = (3, 4)

# Palabras + caracteres 3–4
# USE_WORDS = True
# USE_CHARS = True
# CHAR_NGRAM_RANGE = (3, 3)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ========= UTILIDADES =========

def load_texts_from_folder(folder):
    print(f"[INFO] Cargando textos desde: {folder}")
    texts = []
    for f in os.listdir(folder):
        if f.endswith(".txt"):
            path = os.path.join(folder, f)
            print(f"  - leyendo {f}")
            with open(path, encoding="utf-8") as file:
                texts.append(file.read())
    print(f"[INFO] Total textos cargados: {len(texts)}")
    return texts


def segment_text(text, size):
    words = text.split()
    segments = []
    for i in range(0, len(words), size):
        chunk = words[i:i + size]
        if len(chunk) == size:
            segments.append(" ".join(chunk))
    return segments


def prepare_corpus(texts, use_segmentation):
    prepared = []
    for i, t in enumerate(texts, 1):
        if use_segmentation:
            segs = segment_text(t, SEGMENT_SIZE)
            print(f"[INFO] Texto {i}: {len(segs)} segmentos")
            prepared.extend(segs)
        else:
            prepared.append(t)
    print(f"[INFO] Total muestras tras preparación: {len(prepared)}")
    return prepared


# ========= VECTORIZACIÓN UNIFICADA =========

def unified_vectorize(train_texts, test_texts=None):

    # print(f"[VECT] Vectorizando {len(train_texts)} textos")
    print(f"[VECT] USE_WORDS={USE_WORDS} | USE_CHARS={USE_CHARS}")

    matrices_train = []
    matrices_test = []

    if USE_WORDS:
        print(f"[VECT]  - word ngrams {WORD_NGRAM_RANGE}")
        word_vect = CountVectorizer(
            analyzer="word",
            ngram_range=WORD_NGRAM_RANGE,
            lowercase=True
        )

        Xw_train = word_vect.fit_transform(train_texts)
        Xw_train = normalize(Xw_train, norm="l1")
        matrices_train.append(Xw_train.toarray())

        if test_texts is not None:
            Xw_test = word_vect.transform(test_texts)
            Xw_test = normalize(Xw_test, norm="l1")
            matrices_test.append(Xw_test.toarray())

    if USE_CHARS:
        print(f"[VECT]  - char ngrams {CHAR_NGRAM_RANGE}")
        char_vect = CountVectorizer(
            analyzer="char",
            ngram_range=CHAR_NGRAM_RANGE,
            lowercase=True
        )

        Xc_train = char_vect.fit_transform(train_texts)
        Xc_train = normalize(Xc_train, norm="l1")
        matrices_train.append(Xc_train.toarray())

        if test_texts is not None:
            Xc_test = char_vect.transform(test_texts)
            Xc_test = normalize(Xc_test, norm="l1")
            matrices_test.append(Xc_test.toarray())

    X_train = np.hstack(matrices_train)

    if test_texts is None:
        return X_train

    X_test = np.hstack(matrices_test)

    return X_train, X_test

def build_profile(X):
    return np.mean(X, axis=0)


# ========= O1 – VALIDACIÓN INTERNA =========

def o1_internal_validation(texts):
    print("[O1] Iniciando validación interna")
    distances = []

    for i in range(len(texts)):
        print(f"[O1] Iteración {i+1}/{len(texts)}")
        train = texts[:i] + texts[i+1:]
        test = [texts[i]]

        X_train, X_test = unified_vectorize(train, test)
        profile = build_profile(X_train)

        d = cosine_distances(
            profile.reshape(1, -1),
            X_test.reshape(1, -1)
        )[0][0]

        distances.append(d)

    print("[O1] Validación interna completada")
    return distances


def o1_unknown(candidate_texts, unknown_text):
    print("[O1] Calculando distancia del Dialogus")
    X_train, X_test = unified_vectorize(candidate_texts, [unknown_text])
    profile = build_profile(X_train)

    d = cosine_distances(
        profile.reshape(1, -1),
        X_test.reshape(1, -1)
    )[0][0]

    return d


# ========= O2 – GENERAL IMPOSTERS =========

def o2_iteration(candidate_texts, imposters, unknown_text):
    k = min(5, len(candidate_texts), len(imposters))

    cand_sample = random.sample(candidate_texts, k)
    imp_sample = random.sample(imposters, k)

    train_texts = cand_sample + imp_sample
    X_train, X_test = unified_vectorize(train_texts, [unknown_text])

    cand_profile = build_profile(X_train[:k])
    imp_profile = build_profile(X_train[k:])

    unknown_vec = X_test[0]

    d_cand = cosine_distances(
        cand_profile.reshape(1, -1),
        unknown_vec.reshape(1, -1)
    )[0][0]

    d_imp = cosine_distances(
        imp_profile.reshape(1, -1),
        unknown_vec.reshape(1, -1)
    )[0][0]

    return d_cand < d_imp


def run_o2(candidate_texts, imposters, unknown_text, iterations):
    print(f"[O2] Iniciando General Imposters ({iterations} iteraciones)")
    wins = 0

    for i in range(iterations):
        if i % 50 == 0:
            print(f"[O2] Iteración {i}/{iterations}")
        if o2_iteration(candidate_texts, imposters, unknown_text):
            wins += 1

    print("[O2] General Imposters completado")
    return wins / iterations


# ========= VISUALIZACIONES =========

def plot_o1_boxplot(internal_distances, dialogus_distance):
    print("[PLOT] Guardando boxplot O1")
    plt.figure(figsize=(6, 4))
    plt.boxplot(internal_distances)
    plt.scatter(1, dialogus_distance, color="red", zorder=3, label="Dialogus")
    plt.ylabel("Cosine distance")
    plt.title("O1 – Tácito (validación interna) vs Dialogus")
    plt.legend()
    plt.tight_layout()
    plt.savefig("o1_tacitus_dialogus_boxplot.jpg", dpi=300)
    plt.close()


def plot_o1_histogram(internal_distances, dialogus_distance):
    print("[PLOT] Guardando histograma O1")
    plt.figure(figsize=(6, 4))
    plt.hist(internal_distances, bins=15, alpha=0.7, label="Tácito interno")
    plt.axvline(dialogus_distance, color="red", linestyle="--", label="Dialogus")
    plt.xlabel("Cosine distance")
    plt.ylabel("Frecuencia")
    plt.title("O1 – Distribución de distancias")
    plt.legend()
    plt.tight_layout()
    plt.savefig("o1_tacitus_dialogus_histogram.jpg", dpi=300)
    plt.close()


def plot_o2_score(score):
    print("[PLOT] Guardando gráfico O2")
    plt.figure(figsize=(4, 4))
    plt.bar(["Tácito"], [score])
    plt.ylim(0, 1)
    plt.ylabel("O2 probability")
    plt.title("O2 – Verificación de autoría")
    plt.tight_layout()
    plt.savefig("o2_tacitus_dialogus.jpg", dpi=300)
    plt.close()


# ========= MAIN =========

if __name__ == "__main__":

    print("[MAIN] Cargando corpus")
    tacitus_raw = load_texts_from_folder("corpus/tacitus_all/")
    imposters_raw = load_texts_from_folder("corpus/imposters/")
    dialogus = open("corpus/dialogus.txt", encoding="utf-8").read()

    print("[MAIN] Preparando corpus")
    tacitus_texts = prepare_corpus(tacitus_raw, USE_SEGMENTATION)
    imposters_texts = prepare_corpus(imposters_raw, USE_SEGMENTATION)

    print("[MAIN] Ejecutando O1")
    internal_distances = o1_internal_validation(tacitus_texts)
    dialogus_distance = o1_unknown(tacitus_texts, dialogus)

    plot_o1_boxplot(internal_distances, dialogus_distance)
    plot_o1_histogram(internal_distances, dialogus_distance)

    print("[MAIN] Ejecutando O2")
    o2_score = run_o2(
        tacitus_texts,
        imposters_texts,
        dialogus,
        O2_ITERATIONS
    )

    plot_o2_score(o2_score)

    print("\n=== RESULTADOS ===")
    print("O1 Dialogus distance:", dialogus_distance)
    print("O1 internal mean/std:", np.mean(internal_distances), np.std(internal_distances))
    print("O2 score:", o2_score)
