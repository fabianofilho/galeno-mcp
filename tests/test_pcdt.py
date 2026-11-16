"""Tools: consulta, resumo e conferência da citação."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import duckdb
import httpx
import pytest
import respx

from galeno_mcp.llm.resumir import conferir_citacao
from galeno_mcp.mcp_server.tools.pcdt import TETO_TEXTO, buscar_pcdt, resumir_conduta
from galeno_mcp.store.db import aplicar_schema, conectar, reindexar_fts
from galeno_mcp.store.queries import buscar_condicao, gravar, por_identificador

ENDPOINT = "http://llm-de-teste/v1"
MODELO = "modelo-de-teste"
TEXTO = "3 TRATAMENTO. Primeira linha: fármaco A, 500 mg, duas vezes ao dia, por 30 dias."


def _registro(**campos: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "identificador": "asma",
        "condicao": "Asma",
        "status": "Aprovado*",
        "portaria": "Portaria SECTICS/MS nº 1 - 01/01/2026",
        "data_portaria": date(2026, 1, 1),
        "url_pdf": "https://www.gov.br/conitec/pcdt_asma.pdf",
        "url_resumido": None,
        "texto_completo": TEXTO,
        "secoes_json": json.dumps({"tratamento": TEXTO}),
        "vigente": True,
        "substituido_por": None,
        "extracao_incompleta": False,
    }
    base.update(campos)
    return base


def _chat(conteudo: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": conteudo}}]})


def test_gravar_e_idempotente(db: duckdb.DuckDBPyConnection) -> None:
    assert gravar(db, [_registro()]) == (1, 0)
    assert gravar(db, [_registro()]) == (0, 1)


def test_busca_ignora_acento(db: duckdb.DuckDBPyConnection) -> None:
    gravar(db, [_registro(identificador="acidentes ofídicos", condicao="Acidentes Ofídicos")])
    assert len(buscar_condicao(db, "ofidicos")) == 1


def test_busca_prioriza_nome_curto(db: duckdb.DuckDBPyConnection) -> None:
    gravar(
        db,
        [
            _registro(identificador="asma", condicao="Asma"),
            _registro(identificador="asma grave refrataria", condicao="Asma Grave Refratária"),
        ],
    )
    assert [r["condicao"] for r in buscar_condicao(db, "asma")][0] == "Asma"


def test_por_identificador(db: duckdb.DuckDBPyConnection) -> None:
    gravar(db, [_registro()])
    assert por_identificador(db, "asma") is not None
    assert por_identificador(db, "inexistente") is None


# --- conferência da citação ------------------------------------------------


def test_citacao_literal_confere() -> None:
    assert conferir_citacao("Primeira linha: fármaco A, 500 mg", TEXTO) is True


def test_citacao_parafraseada_nao_confere() -> None:
    """O modelo às vezes 'cita' parafraseando: isso precisa ser detectado."""
    assert conferir_citacao("O protocolo recomenda o fármaco A como primeira opção", TEXTO) is False


def test_citacao_vazia_nao_confere() -> None:
    assert conferir_citacao(None, TEXTO) is False
    assert conferir_citacao("   ", TEXTO) is False


def test_citacao_ignora_espacamento() -> None:
    assert conferir_citacao("Primeira   linha:\n fármaco A", TEXTO) is True


# --- tools -----------------------------------------------------------------


async def test_base_inexistente_avisa(caminho_db: str) -> None:
    resposta = await buscar_pcdt("asma", caminho_db=caminho_db)
    assert resposta.total == 0
    assert resposta.aviso is not None and "sincronizada" in resposta.aviso


async def test_consulta_traz_link_oficial(caminho_db: str) -> None:
    with conectar(caminho_db) as conexao:
        gravar(conexao, [_registro()])
    resposta = await buscar_pcdt("asma", caminho_db=caminho_db)
    assert resposta.total == 1
    assert resposta.resultados[0].url_pdf is not None
    assert resposta.resultados[0].secoes_disponiveis == ["tratamento"]


@respx.mock
async def test_resumo_marca_citacao_confirmada(caminho_db: str) -> None:
    with conectar(caminho_db) as conexao:
        gravar(conexao, [_registro()])
    respx.get(f"{ENDPOINT}/models").mock(return_value=httpx.Response(200, json={"data": []}))
    respx.post(f"{ENDPOINT}/chat/completions").mock(
        return_value=_chat(
            '{"resumo": "Fármaco A por 30 dias.", "secao_origem": "3 TRATAMENTO",'
            ' "citacao_literal": "Primeira linha: fármaco A, 500 mg"}'
        )
    )
    resposta = await resumir_conduta(
        "asma",
        "adulto sem comorbidade",
        caminho_db=caminho_db,
        qwen_endpoint=ENDPOINT,
        qwen_model=MODELO,
    )
    assert resposta.resumo is not None
    assert resposta.resumo.citacao_confere is True
    assert resposta.aviso is not None and "não substitui" in resposta.aviso


@respx.mock
async def test_resumo_denuncia_citacao_inventada(caminho_db: str) -> None:
    """Se a citação não existe no protocolo, o aviso precisa dizer isso."""
    with conectar(caminho_db) as conexao:
        gravar(conexao, [_registro()])
    respx.get(f"{ENDPOINT}/models").mock(return_value=httpx.Response(200, json={"data": []}))
    respx.post(f"{ENDPOINT}/chat/completions").mock(
        return_value=_chat(
            '{"resumo": "Usar fármaco B.", "secao_origem": "3", '
            '"citacao_literal": "fármaco B, 200 mg ao dia"}'
        )
    )
    resposta = await resumir_conduta(
        "asma", "adulto", caminho_db=caminho_db, qwen_endpoint=ENDPOINT, qwen_model=MODELO
    )
    assert resposta.resumo is not None
    assert resposta.resumo.citacao_confere is False
    assert resposta.aviso is not None and "NÃO foi encontrada" in resposta.aviso


@respx.mock
async def test_llm_fora_do_ar_devolve_o_link(caminho_db: str) -> None:
    with conectar(caminho_db) as conexao:
        gravar(conexao, [_registro()])
    respx.get(f"{ENDPOINT}/models").mock(side_effect=httpx.ConnectError("recusado"))
    resposta = await resumir_conduta(
        "asma", "adulto", caminho_db=caminho_db, qwen_endpoint=ENDPOINT, qwen_model=MODELO
    )
    assert resposta.resumo is None
    assert resposta.url_pdf is not None
    assert resposta.aviso is not None and "PDF continua acessível" in resposta.aviso


async def test_protocolo_sem_texto_avisa(caminho_db: str) -> None:
    with conectar(caminho_db) as conexao:
        gravar(conexao, [_registro(texto_completo=None, secoes_json=None)])
    resposta = await resumir_conduta(
        "asma", "adulto", caminho_db=caminho_db, qwen_endpoint=ENDPOINT, qwen_model=MODELO
    )
    assert resposta.resumo is None
    assert resposta.aviso is not None and "ainda não foi coletado" in resposta.aviso


async def test_identificador_inexistente(caminho_db: str) -> None:
    with conectar(caminho_db) as conexao:
        gravar(conexao, [_registro()])
    resposta = await resumir_conduta(
        "nao-existe", "adulto", caminho_db=caminho_db, qwen_endpoint=ENDPOINT, qwen_model=MODELO
    )
    assert resposta.aviso is not None and "buscar_pcdt" in resposta.aviso


# --- fração verificada -----------------------------------------------------


def test_fracao_detecta_costura_de_trechos_reais() -> None:
    """O caso real: frases que existem, mas em partes diferentes do documento."""
    from galeno_mcp.llm.resumir import fracao_verificada

    protocolo = (
        "Primeira linha: fármaco A, 500 mg, duas vezes ao dia. "
        "Muitas páginas de texto no meio do documento. "
        "O diagnóstico precoce é fundamental para definir a dose."
    )
    costura = (
        "Primeira linha: fármaco A, 500 mg, duas vezes ao dia. "
        "O diagnóstico precoce é fundamental para definir a dose."
    )
    assert conferir_citacao(costura, protocolo) is False
    assert fracao_verificada(costura, protocolo) == 1.0


def test_fracao_zero_para_citacao_inventada() -> None:
    from galeno_mcp.llm.resumir import fracao_verificada

    assert fracao_verificada("Frase que não existe em lugar nenhum do documento.", TEXTO) == 0.0


def test_fracao_zero_para_citacao_vazia() -> None:
    from galeno_mcp.llm.resumir import fracao_verificada

    assert fracao_verificada(None, TEXTO) == 0.0


@respx.mock
async def test_aviso_distingue_costura_de_invencao(caminho_db: str) -> None:
    with conectar(caminho_db) as conexao:
        gravar(conexao, [_registro()])
    respx.get(f"{ENDPOINT}/models").mock(return_value=httpx.Response(200, json={"data": []}))
    respx.post(f"{ENDPOINT}/chat/completions").mock(
        return_value=_chat(
            '{"resumo": "x", "secao_origem": "3", '
            '"citacao_literal": "Frase completamente inventada sobre outro assunto."}'
        )
    )
    resposta = await resumir_conduta(
        "asma", "adulto", caminho_db=caminho_db, qwen_endpoint=ENDPOINT, qwen_model=MODELO
    )
    assert resposta.aviso is not None and "NÃO foi encontrada" in resposta.aviso


def test_prompt_carrega_de_dentro_do_pacote() -> None:
    """O prompt precisa vir do pacote instalado, não de um diretório ao lado do repo."""
    from galeno_mcp.llm.resumir import renderizar_prompt

    assert "citacao_literal" in renderizar_prompt()


# --- marcadores de omissão na citação ----------------------------------------

TRECHO_REAL = "Primeira linha: fármaco A, 500 mg, duas vezes ao dia, por 30 dias."


def test_elipse_nas_pontas_nao_derruba_a_citacao() -> None:
    """Bug real: '[...] ' + trecho real dava fração 0.5 e 'NÃO encontrada'."""
    from galeno_mcp.llm.resumir import fracao_verificada

    for marcador in ("[...]", "(...)", "...", chr(0x2026)):
        citacao = f"{marcador} {TRECHO_REAL} {marcador}"
        assert conferir_citacao(citacao, TEXTO) is True, marcador
        assert fracao_verificada(citacao, TEXTO) == 1.0, marcador


def test_elipse_no_meio_nao_e_contigua_mas_conta_na_fracao() -> None:
    from galeno_mcp.llm.resumir import fracao_verificada

    protocolo = (
        "Primeira linha: fármaco A, 500 mg, duas vezes ao dia. "
        "Muitas páginas de texto no meio do documento. "
        "O diagnóstico precoce é fundamental para definir a dose."
    )
    citacao = (
        "Primeira linha: fármaco A, 500 mg, duas vezes ao dia [...] "
        "O diagnóstico precoce é fundamental para definir a dose."
    )
    assert conferir_citacao(citacao, protocolo) is False
    assert fracao_verificada(citacao, protocolo) == 1.0


def test_so_elipse_nao_confere() -> None:
    from galeno_mcp.llm.resumir import fracao_verificada

    assert conferir_citacao("[...]", TEXTO) is False
    assert fracao_verificada("...", TEXTO) == 0.0


@respx.mock
async def test_aviso_nao_acusa_citacao_real_com_elipse(caminho_db: str) -> None:
    with conectar(caminho_db) as conexao:
        gravar(conexao, [_registro()])
    respx.get(f"{ENDPOINT}/models").mock(return_value=httpx.Response(200, json={"data": []}))
    respx.post(f"{ENDPOINT}/chat/completions").mock(
        return_value=_chat(
            json.dumps(
                {"resumo": "x", "secao_origem": "3", "citacao_literal": f"[...] {TRECHO_REAL}"}
            )
        )
    )
    resposta = await resumir_conduta(
        "asma", "adulto", caminho_db=caminho_db, qwen_endpoint=ENDPOINT, qwen_model=MODELO
    )
    assert resposta.resumo is not None and resposta.resumo.citacao_confere is True
    assert resposta.aviso is not None and "ATENÇÃO" not in resposta.aviso


async def test_resumir_aceita_id_sem_nota_de_revisao(caminho_db: str) -> None:
    """Bug real: resumir_conduta('asma', ...) não achava 'asma (anexo alterado em ...)'."""
    with conectar(caminho_db) as conexao:
        gravar(
            conexao,
            [_registro(identificador="asma (anexo alterado em 04/09/2026)", texto_completo=None)],
        )
    resposta = await resumir_conduta(
        "asma", "gestante", caminho_db=caminho_db, qwen_endpoint=ENDPOINT, qwen_model=MODELO
    )
    assert resposta.identificador == "asma (anexo alterado em 04/09/2026)"
    assert resposta.aviso is not None and "ainda não foi coletado" in resposta.aviso


# --- sinônimos e origem do resultado -------------------------------------------


def _nomes(db: duckdb.DuckDBPyConnection, termo: str) -> list[str]:
    return [r["identificador"] for r in buscar_condicao(db, termo)]


def _base_com_diabetes(db: duckdb.DuckDBPyConnection) -> None:
    gravar(
        db,
        [
            _registro(identificador=nome.lower(), condicao=nome)
            for nome in (
                "Diabetes Insípido",
                "Diabete Melito Tipo 1",
                "Diabete Melito Tipo 2",
                "Hipertensão Arterial Sistêmica",
                "Hipertensão Pulmonar",
                "Doença Pulmonar Obstrutiva Crônica",
                "Glaucoma",
            )
        ],
    )


def test_diabetes_acha_dm1_e_dm2(db: duckdb.DuckDBPyConnection) -> None:
    """Bug real: 'diabetes' só trazia 'diabetes insípido'."""
    _base_com_diabetes(db)
    assert set(_nomes(db, "diabetes")) == {
        "diabetes insípido",
        "diabete melito tipo 1",
        "diabete melito tipo 2",
    }


def test_diabetes_mellitus_acha_diabete_melito(db: duckdb.DuckDBPyConnection) -> None:
    _base_com_diabetes(db)
    assert _nomes(db, "Diabetes Mellitus") == ["diabete melito tipo 1", "diabete melito tipo 2"]
    assert _nomes(db, "diabetes tipo 2") == ["diabete melito tipo 2"]
    assert _nomes(db, "DM2") == ["diabete melito tipo 2"]


def test_siglas_viram_nome_por_extenso(db: duckdb.DuckDBPyConnection) -> None:
    _base_com_diabetes(db)
    assert _nomes(db, "HAS") == ["hipertensão arterial sistêmica"]
    assert _nomes(db, "dpoc") == ["doença pulmonar obstrutiva crônica"]


def test_palavras_em_qualquer_ordem(db: duckdb.DuckDBPyConnection) -> None:
    _base_com_diabetes(db)
    assert _nomes(db, "pulmonar hipertensão") == ["hipertensão pulmonar"]


async def test_resultado_so_do_texto_vem_marcado_e_com_aviso(caminho_db: str) -> None:
    """Bug real: 'depressão' devolvia Alzheimer e Parkinson como se fossem o PCDT dela."""
    with conectar(caminho_db) as conexao:
        gravar(
            conexao,
            [
                _registro(
                    identificador="doença de parkinson",
                    condicao="Doença de Parkinson",
                    texto_completo="Sintomas não motores incluem depressão e ansiedade.",
                )
            ],
        )
    resposta = await buscar_pcdt("depressão", caminho_db=caminho_db)
    assert resposta.total == 1
    assert resposta.resultados[0].origem == "texto"
    assert resposta.aviso is not None and "apenas citam o termo" in resposta.aviso


async def test_resultado_pelo_nome_vem_marcado_sem_aviso(caminho_db: str) -> None:
    with conectar(caminho_db) as conexao:
        gravar(conexao, [_registro()])
    resposta = await buscar_pcdt("asma", caminho_db=caminho_db)
    assert resposta.resultados[0].origem == "nome"
    assert resposta.aviso is None


def test_busca_casa_no_comeco_da_palavra(db: duckdb.DuckDBPyConnection) -> None:
    """Bug real: 'asma' trazia 'vasculite ... anti-citoplasma de neutrófilos'."""
    gravar(
        db,
        [
            _registro(identificador="asma", condicao="Asma"),
            _registro(
                identificador="vasculite",
                condicao="Vasculite Associada aos Anticorpos Anti-citoplasma de Neutrófilos",
            ),
        ],
    )
    assert _nomes(db, "asma") == ["asma"]
    assert _nomes(db, "citoplasma") == ["vasculite"]


@pytest.mark.asyncio
async def test_total_e_quantos_casam_nao_o_tamanho_da_lista(caminho_db: str) -> None:
    """'diabetes' aparece no texto de 47 protocolos e a resposta dizia total 10."""
    with conectar(caminho_db) as conexao:
        aplicar_schema(conexao)
        conexao.executemany(
            "INSERT INTO protocolos (identificador, condicao, url_pdf, texto_completo, vigente) "
            "VALUES (?, ?, 'u', 'trata de diabetes entre outras coisas', true)",
            [[str(i), f"CONDICAO RARA {i}"] for i in range(20)],
        )

    r = await buscar_pcdt("diabetes", caminho_db=caminho_db, limite=5)

    assert r.total == 20, "o universo que casa, não a página"
    assert r.retornados == 5
    assert r.truncado is True
    assert r.aviso is not None and "20" in r.aviso


@pytest.mark.asyncio
async def test_total_no_texto_nao_conta_protocolo_nao_vigente(caminho_db: str) -> None:
    """Bug real: total vinha diferente do que a busca no texto conseguia devolver.

    contar_no_texto usava LIKE sem o filtro vigente, enquanto buscar_no_texto (FTS com
    fallback LIKE) só olha vigente. Um protocolo revogado que menciona o termo inflava o
    total sem jamais poder aparecer na lista.
    """
    with conectar(caminho_db) as conexao:
        gravar(
            conexao,
            [
                _registro(identificador="d1", condicao="CONDICAO RARA 1", texto_completo=TEXTO),
                _registro(identificador="d2", condicao="CONDICAO RARA 2", texto_completo=TEXTO),
                _registro(
                    identificador="d3",
                    condicao="CONDICAO RARA 3",
                    texto_completo=TEXTO,
                    vigente=False,
                ),
            ],
        )

    r = await buscar_pcdt("fármaco", caminho_db=caminho_db)

    assert r.total == 2, "o revogado nunca aparece na busca, não deveria entrar no total"
    assert r.retornados == 2
    assert r.truncado is False


def _base_protese(caminho_db: str, *, com_fts: bool) -> None:
    """Dois protocolos com 'prótese' e 'mamária' soltas e um com a frase."""
    with conectar(caminho_db) as conexao:
        gravar(
            conexao,
            [
                _registro(
                    identificador="câncer de mama",
                    condicao="Câncer de Mama",
                    texto_completo="Glândula mamária. Na reconstrução, a prótese de silicone.",
                ),
                _registro(
                    identificador="leiomioma de útero",
                    condicao="Leiomioma de Útero",
                    texto_completo="Prótese não se aplica. Tecido mamário e mamária.",
                ),
                _registro(
                    identificador="linfoma anaplásico",
                    condicao="Linfoma Anaplásico de Grandes Células",
                    texto_completo="Associado ao implante de PRÓTESE\nMAMÁRIA texturizada.",
                ),
            ],
        )
        if com_fts:
            assert reindexar_fts(conexao)


@pytest.mark.asyncio
@pytest.mark.parametrize("com_fts", [True, False], ids=["fts", "sem-fts"])
async def test_texto_com_varias_palavras_exige_a_frase(caminho_db: str, com_fts: bool) -> None:
    """Bug real (15/11/2026): 'prótese mamária' casava as palavras soltas, total 91,
    com câncer de mama, leiomioma e hemangioma à frente."""
    _base_protese(caminho_db, com_fts=com_fts)

    r = await buscar_pcdt("prótese mamária", caminho_db=caminho_db, limite=10)

    assert [p.identificador for p in r.resultados] == ["linfoma anaplásico"]
    assert r.total == 1
    assert r.truncado is False


@pytest.mark.asyncio
@pytest.mark.parametrize("com_fts", [True, False], ids=["fts", "sem-fts"])
async def test_texto_de_uma_palavra_continua_achando(caminho_db: str, com_fts: bool) -> None:
    _base_protese(caminho_db, com_fts=com_fts)

    r = await buscar_pcdt("prótese", caminho_db=caminho_db)

    assert r.total == 3
    assert r.retornados == 3


def _base_texto(caminho_db: str, quantos: int) -> None:
    with conectar(caminho_db) as conexao:
        gravar(
            conexao,
            [
                _registro(
                    identificador=f"rara {i}",
                    condicao=f"CONDICAO RARA {i}",
                    texto_completo="trata de diabetes entre outras coisas",
                )
                for i in range(quantos)
            ],
        )
        assert reindexar_fts(conexao)


@pytest.mark.asyncio
async def test_limite_vale_na_busca_por_texto(caminho_db: str) -> None:
    """Bug real (15/11/2026): limite=10 devolvia 5 no texto e o aviso mandava aumentar
    o limite, o que não mudava nada."""
    _base_texto(caminho_db, 30)

    r = await buscar_pcdt("diabetes", caminho_db=caminho_db, limite=10)

    assert r.total == 30
    assert r.retornados == 10
    assert r.truncado is True
    assert r.aviso is not None and "aumente 'limite'" in r.aviso


@pytest.mark.asyncio
async def test_limite_acima_do_teto_do_texto_nao_manda_aumentar(caminho_db: str) -> None:
    _base_texto(caminho_db, 30)

    r = await buscar_pcdt("diabetes", caminho_db=caminho_db, limite=50)

    assert r.retornados == TETO_TEXTO
    assert r.truncado is True
    assert r.aviso is not None
    assert "aumente" not in r.aviso
    assert f"no máximo {TETO_TEXTO}" in r.aviso


@pytest.mark.asyncio
@pytest.mark.parametrize("com_fts", [True, False], ids=["fts", "sem-fts"])
async def test_frase_com_numero_ou_stopword_acha_com_e_sem_fts(
    caminho_db: str, com_fts: bool
) -> None:
    """O índice FTS ignora números e stopwords em inglês; exigir todos os termos dele
    zerava 'tipo 2' no caminho com FTS enquanto o sem FTS achava."""
    with conectar(caminho_db) as conexao:
        gravar(
            conexao,
            [
                _registro(
                    identificador="dislipidemia",
                    condicao="Dislipidemia",
                    texto_completo="Pacientes com diabetes tipo 2 e reposição de vitamina B12.",
                ),
                _registro(
                    identificador="outro",
                    condicao="Outro",
                    texto_completo="Tipo de reação adversa em 2 casos.",
                ),
            ],
        )
        if com_fts:
            assert reindexar_fts(conexao)

    for termo in ("tipo 2", "vitamina b12", "diabetes tipo 2"):
        r = await buscar_pcdt(termo, caminho_db=caminho_db)
        assert [p.identificador for p in r.resultados] == ["dislipidemia"], termo
        assert r.total == 1, termo
