"""Script local — análise emocional de comentários do Reddit."""

import argparse
import os

from analise_core import executar_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Análise emocional e agrupamento por perfil de comentários do Reddit."
    )
    parser.add_argument(
        "--pasta",
        default=os.environ.get("REDDIT_CSV_PASTA", "."),
        help="Pasta com arquivos CSV (ou variável REDDIT_CSV_PASTA)",
    )
    parser.add_argument(
        "--saida",
        default=".",
        help="Pasta de saída dos arquivos gerados",
    )
    parser.add_argument(
        "--idioma",
        choices=["pt", "en", "both", "none"],
        default="both",
        help="Stop words do TF-IDF: both (EN+PT, padrão), pt, en ou none",
    )
    parser.add_argument(
        "--min-palavras",
        type=int,
        default=3,
        help="Mínimo de palavras para classificar (exclui textos mais curtos)",
    )
    parser.add_argument(
        "--manter-curtos",
        action="store_true",
        help="Não exclui comentários curtos nem links (equivale a min-palavras=0)",
    )
    parser.add_argument(
        "--sem-duplicatas",
        action="store_true",
        help="Remove comentários duplicados antes da análise",
    )
    parser.add_argument(
        "--cache",
        default="cache_emocoes.json",
        help="Arquivo de cache das classificações emocionais",
    )
    parser.add_argument(
        "--sem-cache",
        action="store_true",
        help="Desativa o cache de classificações",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Tamanho do batch na classificação emocional",
    )
    args = parser.parse_args()

    min_palavras = 0 if args.manter_curtos else args.min_palavras
    filtrar_links = not args.manter_curtos

    resultado = executar_pipeline(
        args.pasta,
        saida_dir=args.saida,
        idioma_stop=args.idioma,
        remover_duplicatas=args.sem_duplicatas,
        min_palavras=min_palavras,
        filtrar_links=filtrar_links,
        cache_path=None if args.sem_cache else args.cache,
        batch_size=args.batch_size,
    )

    print("Arquivos gerados:")
    for chave in ("csv", "palavras_txt", "resumo_txt", "stats_csv", "excluidos_csv"):
        if chave in resultado:
            print(f"  - {resultado[chave]}")


if __name__ == "__main__":
    main()
