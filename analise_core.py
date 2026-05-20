"""Pipeline compartilhado: análise emocional e agrupamento por perfil."""

import hashlib
import json
import logging
import os
import re
from glob import glob
from pathlib import Path
from typing import Optional, Set, Union

import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import pipeline

logger = logging.getLogger(__name__)

LABELS_EMOCIONAIS = [
    "Cético Racional",
    "Perdido Espiritual",
    "Curioso Cético",
    "Impulsivo Esperançoso",
]

COLUNAS_EXTRA = ["Author", "Subreddit", "Post", "Score"]
MODEL_NAME = "facebook/bart-large-mnli"
MAX_CHARS = 2000
BATCH_SIZE = 8
TOP_PALAVRAS = 10
EXEMPLOS_POR_PERFIL = 8
MIN_PALAVRAS_PADRAO = 3

# Ruído típico de URLs e markup no Reddit
STOP_WORDS_EXTRA = frozenset(
    """
    http https www com org net amp gt lt nbsp reddit wwwreddit
    """.split()
)

# sklearn recente só inclui stop words embutidas em inglês
STOP_WORDS_EN = "english"
STOP_WORDS_PT = frozenset(
    """
    a à às ao aos aquela aquelas aquele aqueles aquilo as às assim através
    bem boa boas bom bons com como contra contudo da das de dela delas dele
    deles depois dessa dessas desse desses desta destas deste destes deve
    devem deverá deverão disse disso do dos e é ela elas ele eles em entre
    era eram essa essas esse esses esta estas este estes eu foi foram há
    isso isto já lhe lhes mais mas me mesmo meu meus minha minhas muito
    na nas não nos nossa nossas nosso nossos num numa nunca o os ou para
    pela pelas pelo pelos por porque qual quando que quem se sem sempre
    ser será serão seu seus só somos sou sua suas também te tem têm teu
    teus tu tua tuas um uma umas uns você vocês vos
    """.split()
)

STOP_WORDS_MAP = {
    "pt": "pt",
    "en": "en",
    "both": "both",
    "none": "none",
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_APENAS_URL_RE = re.compile(
    r"^\s*(https?://\S+|www\.\S+|\[.*?\]\(https?://[^\)]+\))\s*$",
    re.IGNORECASE,
)
_PALAVRA_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9']+")

COLUNAS_ALIASES = {
    "Comment Author": "Author",
}


def _configurar_logging():
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )


def _hash_comentario(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _device_pipeline():
    return 0 if torch.cuda.is_available() else -1


def _truncar_texto(texto: str, max_chars: int = MAX_CHARS) -> str:
    texto = str(texto).strip()
    if len(texto) <= max_chars:
        return texto
    return texto[:max_chars]


def _garantir_nltk_stopwords():
    import nltk

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def _stopwords_nltk(idioma: str) -> Set[str]:
    from nltk.corpus import stopwords

    _garantir_nltk_stopwords()
    if idioma == "english":
        return set(stopwords.words("english"))
    if idioma == "portuguese":
        return set(stopwords.words("portuguese"))
    return set()


def _resolver_stop_words(idioma: str) -> Optional[Union[str, list]]:
    """Retorna stop words para TfidfVectorizer (str 'english', lista ou None)."""
    chave = (idioma or "both").lower()
    if chave not in STOP_WORDS_MAP:
        raise ValueError(
            f"Idioma '{idioma}' inválido. Use: {', '.join(STOP_WORDS_MAP)}"
        )
    if chave == "none":
        return None

    lista: Set[str] = set(STOP_WORDS_EXTRA)

    if chave in ("en", "both"):
        try:
            lista |= _stopwords_nltk("english")
        except Exception as e:
            logger.warning("NLTK inglês indisponível: %s", e)
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

            lista |= set(ENGLISH_STOP_WORDS)

    if chave in ("pt", "both"):
        try:
            lista |= _stopwords_nltk("portuguese")
        except Exception as e:
            logger.warning("NLTK português indisponível, usando lista local: %s", e)
            lista |= STOP_WORDS_PT
        else:
            lista |= STOP_WORDS_PT

    return sorted(lista)


def _texto_sem_urls(texto: str) -> str:
    return _URL_RE.sub(" ", texto)


def contar_palavras_significativas(texto: str) -> int:
    """Conta tokens alfabéticos após remover URLs."""
    limpo = _texto_sem_urls(str(texto))
    return len(_PALAVRA_RE.findall(limpo))


def eh_apenas_link(texto: str) -> bool:
    """True se o comentário for só URL ou markup de link."""
    t = str(texto).strip()
    if not t:
        return True
    if _APENAS_URL_RE.match(t):
        return True
    sem_urls = _texto_sem_urls(t).strip()
    return not sem_urls or contar_palavras_significativas(t) == 0


def filtrar_comentarios(
    df: pd.DataFrame,
    *,
    min_palavras: int = MIN_PALAVRAS_PADRAO,
    filtrar_links: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa comentários válidos dos inválidos (curtos ou só link).
    Retorna (df_validos, df_excluidos).
    """
    df = df.copy()
    motivos = []

    for texto in df["comentario"]:
        n_palavras = contar_palavras_significativas(texto)
        if filtrar_links and eh_apenas_link(texto):
            motivos.append("link_apenas")
        elif n_palavras < min_palavras:
            motivos.append(f"poucas_palavras ({n_palavras})")
        else:
            motivos.append(None)

    df["motivo_exclusao"] = motivos
    excluidos = df[df["motivo_exclusao"].notna()].reset_index(drop=True)
    validos = (
        df[df["motivo_exclusao"].isna()]
        .drop(columns=["motivo_exclusao"])
        .reset_index(drop=True)
    )

    if len(excluidos):
        logger.info(
            "Comentários excluídos (curtos/links): %d de %d",
            len(excluidos),
            len(df),
        )

    return validos, excluidos


def carregar_comentarios(pasta=".", remover_duplicatas: bool = False) -> pd.DataFrame:
    """Carrega CSVs com coluna Comment e colunas extras do scraper PRAW, se existirem."""
    padrao = os.path.join(pasta, "*.csv")
    arquivos = glob(padrao)
    if not arquivos:
        raise ValueError(f"Nenhum CSV encontrado em: {pasta}")

    partes = []
    for arquivo in arquivos:
        df = pd.read_csv(arquivo)
        if "Comment" not in df.columns:
            continue

        colunas = ["Comment"] + [c for c in COLUNAS_EXTRA if c in df.columns]
        for alias, destino in COLUNAS_ALIASES.items():
            if alias in df.columns and destino not in colunas:
                colunas.append(alias)
        trecho = df[colunas].copy()
        trecho = trecho.rename(
            columns={"Comment": "comentario", **COLUNAS_ALIASES}
        )
        partes.append(trecho)

    if not partes:
        raise ValueError("Nenhum arquivo com a coluna 'Comment' foi encontrado.")

    df_total = pd.concat(partes, ignore_index=True)
    df_total.dropna(subset=["comentario"], inplace=True)
    df_total["comentario"] = df_total["comentario"].astype(str)
    df_total = df_total[df_total["comentario"].str.strip() != ""]

    if remover_duplicatas:
        antes = len(df_total)
        df_total = df_total.drop_duplicates(subset=["comentario"]).reset_index(drop=True)
        logger.info("Duplicatas removidas: %d -> %d", antes, len(df_total))

    return df_total


def _carregar_cache(caminho_cache: Path) -> dict:
    if not caminho_cache.exists():
        return {}
    try:
        with open(caminho_cache, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Cache inválido (%s), recriando: %s", caminho_cache, e)
        return {}


def _salvar_cache(caminho_cache: Path, cache: dict) -> None:
    caminho_cache.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho_cache, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def detectar_emocoes(
    df: pd.DataFrame,
    *,
    batch_size: int = BATCH_SIZE,
    cache_path: Optional[str] = "cache_emocoes.json",
    model_name: str = MODEL_NAME,
) -> pd.DataFrame:
    """Classifica comentários em perfis emocionais (batch, GPU, cache)."""
    df = df.copy()
    textos = [_truncar_texto(t) for t in df["comentario"]]
    hashes = [_hash_comentario(t) for t in textos]

    caminho_cache = Path(cache_path) if cache_path else None
    cache = _carregar_cache(caminho_cache) if caminho_cache else {}

    indices_pendentes = [
        i for i, h in enumerate(hashes) if h not in cache
    ]

    if indices_pendentes:
        device = _device_pipeline()
        logger.info(
            "Classificando %d comentários (batch=%d, device=%s)...",
            len(indices_pendentes),
            batch_size,
            "cuda" if device == 0 else "cpu",
        )
        classificador = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device,
        )

        textos_pendentes = [textos[i] for i in indices_pendentes]
        for inicio in tqdm(
            range(0, len(textos_pendentes), batch_size),
            desc="Analisando emoções",
        ):
            lote_idx = indices_pendentes[inicio : inicio + batch_size]
            lote_textos = [textos[i] for i in lote_idx]
            try:
                resultados = classificador(
                    lote_textos,
                    candidate_labels=LABELS_EMOCIONAIS,
                    batch_size=len(lote_textos),
                )
                if isinstance(resultados, dict):
                    resultados = [resultados]

                for idx, resultado in zip(lote_idx, resultados):
                    cache[hashes[idx]] = {
                        "perfil_emocional": resultado["labels"][0],
                        "score_emocional": float(resultado["scores"][0]),
                    }
            except Exception as e:
                logger.error("Erro no batch %s-%s: %s", inicio, inicio + batch_size, e)
                for idx in lote_idx:
                    cache[hashes[idx]] = {
                        "perfil_emocional": "Erro",
                        "score_emocional": 0.0,
                    }

        if caminho_cache:
            _salvar_cache(caminho_cache, cache)

    perfis = []
    scores = []
    for h in hashes:
        entrada = cache.get(h, {"perfil_emocional": "Erro", "score_emocional": 0.0})
        perfis.append(entrada["perfil_emocional"])
        scores.append(entrada["score_emocional"])

    df["perfil_emocional"] = perfis
    df["score_emocional"] = scores
    df = df[df["perfil_emocional"] != "Erro"].reset_index(drop=True)
    return df


def vetorizar_textos(df: pd.DataFrame, idioma_stop: str = "both"):
    """TF-IDF global para palavras-chave e visualização."""
    stop_words = _resolver_stop_words(idioma_stop)
    kwargs = {"max_features": 1000, "token_pattern": r"(?u)\b[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ0-9']*\b"}
    if isinstance(stop_words, list):
        kwargs["stop_words"] = stop_words
    elif stop_words is not None:
        kwargs["stop_words"] = stop_words
    vectorizer = TfidfVectorizer(**kwargs)
    X = vectorizer.fit_transform(df["comentario"])
    return X, vectorizer


def reduzir_dimensao(X, n_components: int = 2):
    """Redução 2D em matriz esparsa (TruncatedSVD)."""
    n_components = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
    if n_components < 1:
        return None
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(X)


def palavras_por_perfil(df: pd.DataFrame, X, vectorizer, top_n: int = TOP_PALAVRAS):
    """Top termos TF-IDF agregados dentro de cada perfil emocional."""
    features = vectorizer.get_feature_names_out()
    resultado = {}

    for perfil in sorted(df["perfil_emocional"].unique()):
        indices = df.index[df["perfil_emocional"] == perfil].tolist()
        if not indices:
            continue
        submatrix = X[indices]
        soma = submatrix.sum(axis=0).A1
        min_freq = 0.5
        candidatos = [
            (i, float(soma[i]))
            for i in range(len(soma))
            if soma[i] >= min_freq
        ]
        candidatos.sort(key=lambda x: x[1], reverse=True)
        resultado[perfil] = [(features[i], freq) for i, freq in candidatos[:top_n]]

    return resultado


def agregar_por_perfil(df: pd.DataFrame) -> pd.DataFrame:
    """Estatísticas por perfil emocional."""
    total = len(df)
    linhas = []
    for perfil, grupo in df.groupby("perfil_emocional"):
        linhas.append(
            {
                "perfil_emocional": perfil,
                "quantidade": len(grupo),
                "percentual": round(100 * len(grupo) / total, 2) if total else 0,
                "score_medio": round(grupo["score_emocional"].mean(), 4),
            }
        )
    return pd.DataFrame(linhas).sort_values("quantidade", ascending=False)


def gerar_resumo_chatgpt(
    df: pd.DataFrame,
    palavras: dict,
    stats: pd.DataFrame,
    exemplos_por_perfil: int = EXEMPLOS_POR_PERFIL,
) -> str:
    """Resumo estruturado para uso com LLM."""
    linhas = ["# Resumo por perfil emocional\n"]

    for _, row in stats.iterrows():
        perfil = row["perfil_emocional"]
        linhas.append(f"\n## {perfil}")
        linhas.append(f"- Comentários: {int(row['quantidade'])} ({row['percentual']}%)")
        linhas.append(f"- Confiança média: {row['score_medio']}")

        if perfil in palavras:
            termos = ", ".join(f"{p} ({f:.1f})" for p, f in palavras[perfil][:10])
            linhas.append(f"- Termos frequentes: {termos}")

        grupo = df[df["perfil_emocional"] == perfil].nlargest(
            exemplos_por_perfil, "score_emocional"
        )
        if "Subreddit" in df.columns:
            subs = grupo["Subreddit"].dropna().value_counts().head(5)
            if not subs.empty:
                linhas.append(
                    "- Subreddits frequentes: "
                    + ", ".join(f"{s} ({c})" for s, c in subs.items())
                )

        linhas.append("- Exemplos de comentários:")
        for i, (_, r) in enumerate(grupo.iterrows(), 1):
            texto = _truncar_texto(r["comentario"], 300).replace("\n", " ")
            linhas.append(f"  {i}. [{r['score_emocional']:.2f}] {texto}")

    return "\n".join(linhas) + "\n"


def salvar_palavras_por_perfil(palavras: dict, caminho: str) -> None:
    with open(caminho, "w", encoding="utf-8") as f:
        for perfil, termos in palavras.items():
            f.write(f"\nPerfil: {perfil}\n")
            for palavra, freq in termos:
                f.write(f"- {palavra}: {freq:.1f}\n")


def executar_pipeline(
    pasta: str = ".",
    *,
    saida_dir: str = ".",
    idioma_stop: str = "both",
    remover_duplicatas: bool = False,
    min_palavras: int = MIN_PALAVRAS_PADRAO,
    filtrar_links: bool = True,
    cache_path: Optional[str] = "cache_emocoes.json",
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Executa o pipeline completo e grava os arquivos de saída."""
    _configurar_logging()
    saida = Path(saida_dir)
    saida.mkdir(parents=True, exist_ok=True)

    df = carregar_comentarios(pasta, remover_duplicatas=remover_duplicatas)
    logger.info("Comentários carregados: %d", len(df))

    df, excluidos = filtrar_comentarios(
        df, min_palavras=min_palavras, filtrar_links=filtrar_links
    )
    if excluidos.empty:
        excluidos_path = None
    else:
        excluidos_path = saida / "comentarios_excluidos.csv"
        excluidos.to_csv(excluidos_path, index=False)

    if df.empty:
        raise ValueError(
            "Nenhum comentário válido após filtrar textos curtos e links. "
            "Ajuste --min-palavras ou use --manter-curtos."
        )

    df = detectar_emocoes(df, batch_size=batch_size, cache_path=cache_path)
    logger.info("Comentários classificados: %d", len(df))

    X, vectorizer = vetorizar_textos(df, idioma_stop=idioma_stop)
    coords = reduzir_dimensao(X)
    if coords is not None:
        df["svd_x"], df["svd_y"] = coords.T
    else:
        df["svd_x"], df["svd_y"] = 0.0, 0.0

    palavras = palavras_por_perfil(df, X, vectorizer)
    stats = agregar_por_perfil(df)
    resumo = gerar_resumo_chatgpt(df, palavras, stats)

    csv_path = saida / "saida_analise_chatgpt.csv"
    palavras_path = saida / "palavras_por_perfil.txt"
    resumo_path = saida / "resumo_perfis_chatgpt.txt"
    stats_path = saida / "estatisticas_perfis.csv"

    df.to_csv(csv_path, index=False)
    salvar_palavras_por_perfil(palavras, str(palavras_path))
    with open(resumo_path, "w", encoding="utf-8") as f:
        f.write(resumo)
    stats.to_csv(stats_path, index=False)

    logger.info("Arquivos gerados em %s", saida.resolve())
    resultado = {
        "df": df,
        "stats": stats,
        "palavras": palavras,
        "csv": str(csv_path),
        "palavras_txt": str(palavras_path),
        "resumo_txt": str(resumo_path),
        "stats_csv": str(stats_path),
        "excluidos": excluidos,
    }
    if excluidos_path:
        resultado["excluidos_csv"] = str(excluidos_path)
    return resultado
