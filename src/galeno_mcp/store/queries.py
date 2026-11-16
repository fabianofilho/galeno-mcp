"""Queries sobre a base de protocolos."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

import duckdb

from galeno_mcp.nomes import chave_nome, termos_de_busca

logger = logging.getLogger(__name__)

_COLUNAS_V1 = (
    "identificador, condicao, status, portaria, data_portaria, url_pdf, url_resumido, "
    "texto_completo, secoes_json, vigente, substituido_por, extracao_incompleta"
)


def _colunas(conexao: duckdb.DuckDBPyConnection) -> str:
    """Colunas do SELECT, tolerando uma base da versão 1 aberta só para leitura.

    O servidor MCP abre a base em read_only e não migra nada; até o próximo sync
    a base pode não ter ``nota_atualizacao``.
    """
    tem_nota = conexao.execute(
        "SELECT count(*) FROM information_schema.columns "
        "WHERE table_name = 'protocolos' AND column_name = 'nota_atualizacao'"
    ).fetchone()
    if tem_nota and tem_nota[0]:
        return f"{_COLUNAS_V1}, nota_atualizacao"
    return f"{_COLUNAS_V1}, NULL AS nota_atualizacao"


def _para_dicts(resultado: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    colunas = [d[0] for d in resultado.description or []]
    return [dict(zip(colunas, linha, strict=True)) for linha in resultado.fetchall()]


def buscar_condicao(
    conexao: duckdb.DuckDBPyConnection,
    termo: str,
    *,
    limite: int = 10,
    apenas_vigentes: bool = True,
) -> list[dict[str, Any]]:
    """Busca por doença ou condição no nome do PCDT, tolerante a grafia.

    Ignora acento e caixa, exige que todas as palavras do termo apareçam no nome
    em qualquer ordem, sempre no começo de uma palavra ("asma" não casa com
    "citoplasma"), troca siglas comuns pelo nome por extenso ("HAS", "DPOC",
    "DM2") e aceita grafias diferentes ("diabetes mellitus" acha "diabete
    melito tipo 1" e "tipo 2"). Ordena pelo tamanho do nome, então o PCDT da
    condição vem antes de nomes longos que apenas a mencionam.
    """
    grupos = termos_de_busca(termo)
    if not grupos:
        return []
    condicoes: list[str] = []
    parametros: list[Any] = []
    for grafias in grupos:
        condicoes.append(
            "("
            + " OR ".join("regexp_matches(strip_accents(lower(condicao)), ?)" for _ in grafias)
            + ")"
        )
        parametros.extend(f"(^|[^a-z0-9]){re.escape(g)}" for g in grafias)
    filtro = "AND vigente" if apenas_vigentes else ""
    return _para_dicts(
        conexao.execute(
            f"""
            SELECT {_colunas(conexao)}
            FROM protocolos
            WHERE {" AND ".join(condicoes)}
              {filtro}
            ORDER BY length(condicao), condicao
            LIMIT ?
            """,
            [*parametros, limite],
        )
    )


def contar_condicao(
    conexao: duckdb.DuckDBPyConnection,
    termo: str,
    *,
    apenas_vigentes: bool = True,
) -> int:
    """Quantos protocolos casam pelo nome, ignorando o limite de exibição."""
    grupos = termos_de_busca(termo)
    if not grupos:
        return 0
    condicoes: list[str] = []
    parametros: list[Any] = []
    for grafias in grupos:
        condicoes.append(
            "("
            + " OR ".join("regexp_matches(strip_accents(lower(condicao)), ?)" for _ in grafias)
            + ")"
        )
        parametros.extend(f"(^|[^a-z0-9]){re.escape(g)}" for g in grafias)
    filtro = "AND vigente" if apenas_vigentes else ""
    linha = conexao.execute(
        f"SELECT count(*) FROM protocolos WHERE {' AND '.join(condicoes)} {filtro}",
        parametros,
    ).fetchone()
    return int(linha[0]) if linha else 0


def _palavras(termo: str) -> list[str]:
    """Palavras do termo sem acento e em minúsculas, como o texto é comparado."""
    sem_acento = unicodedata.normalize("NFKD", termo.lower()).encode("ascii", "ignore").decode()
    return re.findall(r"[a-z0-9]+", sem_acento)


def _regex_frase(palavras: list[str]) -> str:
    """Regex das palavras em sequência, cada uma inteira e com plural opcional.

    Entre as palavras aceita espaço, quebra de linha e hífen, que é o que o texto
    extraído do PDF põe entre elas.
    """
    corpo = r"[\s-]+".join(f"{re.escape(p)}s?" for p in palavras)
    return f"(^|[^a-z0-9]){corpo}([^a-z0-9]|$)"


_TEXTO_NORMALIZADO = "strip_accents(lower(coalesce(texto_completo, '')))"


def _filtro_texto(termo: str, *, com_fts: bool) -> tuple[str, list[Any]] | None:
    """WHERE da busca no texto, o mesmo para listar e para contar.

    Bug real (15/11/2026): "prótese mamária" casava as palavras soltas (o FTS do DuckDB
    junta os termos com OU) e devolvia 91 protocolos, câncer de mama e leiomioma à
    frente. Com mais de uma palavra, o termo tem que aparecer como frase no texto. Com
    uma palavra só, o FTS segue sozinho, porque o stemming dele acha "diabético" para
    "diabetes". Sem FTS, a frase vale para qualquer número de palavras.

    Com mais de uma palavra o FTS não entra no filtro, só na ordem: o índice ignora
    números e stopwords em inglês ("a", "as", "no"), então exigir todos os termos dele
    zerava "tipo 2" e "vitamina b12" enquanto o caminho sem FTS os achava.
    """
    palavras = _palavras(termo)
    if not palavras:
        return None
    if com_fts and len(palavras) == 1:
        return (
            "fts_main_protocolos.match_bm25(identificador, ?) IS NOT NULL AND vigente",
            [termo],
        )
    return f"regexp_matches({_TEXTO_NORMALIZADO}, ?) AND vigente", [_regex_frase(palavras)]


def contar_no_texto(conexao: duckdb.DuckDBPyConnection, termo: str) -> int:
    """Quantos protocolos casam no texto completo, pelo mesmo critério de ``buscar_no_texto``.

    Bug real (15/11/2026): esta contagem usava ``LIKE`` sem o filtro ``vigente`` enquanto
    ``buscar_no_texto`` busca por FTS (com fallback pra ``LIKE`` + ``vigente``) só entre os
    vigentes, e as duas contas discordavam. As duas usam ``_filtro_texto``.
    """
    for com_fts in (True, False):
        filtro = _filtro_texto(termo, com_fts=com_fts)
        if filtro is None:
            return 0
        where, parametros = filtro
        try:
            linha = conexao.execute(
                f"SELECT count(*) FROM protocolos WHERE {where}", parametros
            ).fetchone()
            return int(linha[0]) if linha else 0
        except duckdb.Error as erro:
            if not com_fts:
                raise
            logger.debug("FTS indisponível (%s); usando a frase sem FTS", erro)
    return 0


def buscar_no_texto(
    conexao: duckdb.DuckDBPyConnection,
    termo: str,
    *,
    limite: int = 10,
) -> list[dict[str, Any]]:
    """Busca no texto completo, via FTS quando disponível. Critério em ``_filtro_texto``."""
    colunas = _colunas(conexao)
    filtro = _filtro_texto(termo, com_fts=True)
    if filtro is None:
        return []
    where, parametros = filtro
    try:
        return _para_dicts(
            conexao.execute(
                f"""
                SELECT {colunas},
                       fts_main_protocolos.match_bm25(identificador, ?) AS relevancia
                FROM protocolos
                WHERE {where}
                ORDER BY relevancia DESC NULLS LAST, length(condicao)
                LIMIT ?
                """,
                [termo, *parametros, limite],
            )
        )
    except duckdb.Error as erro:
        logger.debug("FTS indisponível (%s); usando a frase sem FTS", erro)

    where, parametros = _filtro_texto(termo, com_fts=False) or ("FALSE", [])
    return _para_dicts(
        conexao.execute(
            f"""
            SELECT {colunas}, NULL AS relevancia
            FROM protocolos
            WHERE {where}
            ORDER BY length(condicao)
            LIMIT ?
            """,
            [*parametros, limite],
        )
    )


def por_identificador(
    conexao: duckdb.DuckDBPyConnection, identificador: str
) -> dict[str, Any] | None:
    """Protocolo pelo identificador exato ou, se não houver, pelo nome sem nota.

    O segundo caminho aceita "asma" quando a base ainda guarda "asma (anexo
    alterado em ...)", e também diferenças de acento e pontuação. Entre vários
    candidatos, prefere o vigente.
    """
    linhas = _para_dicts(
        conexao.execute(
            f"SELECT {_colunas(conexao)} FROM protocolos WHERE identificador = ?", [identificador]
        )
    )
    if linhas:
        return linhas[0]

    alvo = chave_nome(identificador)
    if not alvo:
        return None
    candidatos = conexao.execute(
        "SELECT identificador, condicao FROM protocolos ORDER BY vigente DESC, identificador"
    ).fetchall()
    for ident, condicao in candidatos:
        if alvo in (chave_nome(ident), chave_nome(condicao)):
            return por_identificador(conexao, ident)
    return None


def gravar(conexao: duckdb.DuckDBPyConnection, registros: list[dict[str, Any]]) -> tuple[int, int]:
    """Upsert em massa. Devolve (novos, atualizados)."""
    if not registros:
        return 0, 0

    unicos = {r["identificador"]: r for r in registros}
    linhas = list(unicos.values())
    colunas = list(linhas[0].keys())
    lista = ", ".join(colunas)
    marcadores = ", ".join("?" for _ in colunas)
    atribuicoes = ", ".join(f"{c} = excluded.{c}" for c in colunas if c != "identificador")

    staging = "staging_protocolos"
    conexao.execute(
        f"CREATE OR REPLACE TEMP TABLE {staging} AS SELECT {lista} FROM protocolos LIMIT 0"
    )
    conexao.executemany(
        f"INSERT INTO {staging} ({lista}) VALUES ({marcadores})",
        [[r[c] for c in colunas] for r in linhas],
    )
    contagem = conexao.execute(
        f"""
        SELECT count(*) FROM {staging} s
        WHERE NOT EXISTS (
            SELECT 1 FROM protocolos p WHERE p.identificador = s.identificador
        )
        """
    ).fetchone()
    novos = int(contagem[0]) if contagem else 0

    conexao.execute(
        f"""
        INSERT INTO protocolos ({lista})
        SELECT {lista} FROM {staging}
        ON CONFLICT (identificador) DO UPDATE SET {atribuicoes}, data_coleta = now()
        """
    )
    conexao.execute(f"DROP TABLE {staging}")
    return novos, len(linhas) - novos


def marcar_ausentes_como_substituidos(
    conexao: duckdb.DuckDBPyConnection, identificadores_vistos: list[str]
) -> int:
    """Protocolo que sumiu da listagem deixou de ser vigente.

    Não apaga: a spec pede manter histórico, e um PCDT removido ainda é útil
    para entender o que valia antes.
    """
    if not identificadores_vistos:
        return 0
    marcadores = ", ".join("?" for _ in identificadores_vistos)
    resultado = conexao.execute(
        f"""
        UPDATE protocolos SET vigente = FALSE
        WHERE vigente AND identificador NOT IN ({marcadores})
        """,
        identificadores_vistos,
    )
    linhas = resultado.fetchall()
    return int(linhas[0][0]) if linhas and linhas[0] else 0
