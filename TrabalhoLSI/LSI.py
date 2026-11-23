# importacao de bibliotecas basicas
import math
import re
import unicodedata
import numpy as np

# parametro global de reducao dimensional
k_dimensoes = 3

# acervo de documentos organizados em 3 dominios tematicos
base_documentos = [
    # dominio 1: gastronomia 
    "O chef preparou uma receita deliciosa de bolo de chocolate no forno.",
    "Assar massas e paes no forno exige uma cozinha bem equipada.",
    "Esta receita tradicional de sobremesa doce leva ingredientes frescos.",
    "O chef do restaurante serviu uma comida italiana com molho de tomate.",
    "Confeiteiros preparam tortas, doces e bolos na cozinha para festas.",

    # dominio 2: tecnologia
    "O desenvolvimento de software moderno exige boas praticas de programacao.",
    "Engenheiros escrevem codigo limpo para criar aplicativos e sistemas.",
    "A linguagem Python facilita a criacao de sistemas e analise de dados.",
    "Bancos de dados e sistemas operacionais gerenciam a memoria do computador.",
    "Programadores projetam algoritmos eficientes para processar dados no computador.",

    # dominio 3: esportes e saude
    "A corrida diaria no parque melhora o condicionamento e a saude.",
    "O treino de musculacao fortalece o corpo e aumenta a resistencia.",
    "Atletas seguem rotinas pesadas de exercicio fisico e alimentacao.",
    "Praticar esporte como natacao e futebol faz bem para a saude.",
    "O preparo fisico do atleta depende de treino regular e descanso."
]

# palavras funcionais sem carga semantica para descarte
stopwords_pt = {
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com", "nao",
     "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas",
    "ao", "ele", "das", "seu", "sua", "ou", "quando", "muito", "nos", "ja",
    "eu", "tambem", "so", "pelo", "pela", "ate", "isso", "ela", "entre",
    "sem", "mesmo", "aos", "ter", "seus", "quem", "nas", "me", "esse", "eles",
    "estao", "voce", "tinha", "foram", "essa", "num", "nem", "suas", "meu",
    "minha", "tem", "numa", "pelos", "elas", "havia", "seja", "qual", "sera",
    "sao", "todos", "bem", "otimo", "otima", "esta", "este", "leva", "fazem"
}


# funcoes de pre-processamento textual



def remover_acentos(texto):
    # remove acentos e cedilha usando decomposicao unicode
    forma_decomposta = unicodedata.normalize("NFKD", texto)

    return forma_decomposta.encode("ASCII", "ignore").decode("utf-8")

def normalizar_e_tokenizar(texto):

    # limpa o texto, extrai palavras, filtra stopwords e reduz plurais simples

    texto_limpo = remover_acentos(texto).lower()
    tokens_brutos = re.findall(r"\b[a-z]+\b", texto_limpo)

    tokens_filtrados = []
    for token in tokens_brutos:


        # ignora stopwords e palavras com 2 ou menos letras
        if token in stopwords_pt or len(token) <= 2:
            continue

        # reduz plural regular simples


        if token.endswith("s") and len(token) > 4:
            token = token[:-1]

        if token not in stopwords_pt:
            tokens_filtrados.append(token)

    return tokens_filtrados

def tokenizar_colecao(documentos):



    # aplica a tokenizacao em todos os documentos da base
    colecao_tokenizada = []
    for doc in documentos:
        tokens = normalizar_e_tokenizar(doc)
        colecao_tokenizada.append(tokens)
    return colecao_tokenizada


# funcoes de construcao do vocabulario e matriz termo-documento

def construir_vocabulario(docs_tokenizados):



    # extrai termos unicos ordenados e mapeia cada termo ao seu indice
    conjunto_termos = set()
    for tokens in docs_tokenizados:
        for termo in tokens:
            conjunto_termos.add(termo)
    vocabulario = sorted(list(conjunto_termos))
    termo_para_idx = {}
    for indice, termo in enumerate(vocabulario):
        termo_para_idx[termo] = indice

    return vocabulario, termo_para_idx


def calcular_idf(docs_tokenizados, vocabulario, termo_para_idx):



    # calcula o peso global idf suavizado de cada termo

    m_termos = len(vocabulario)

    n_docs = len(docs_tokenizados)

    # conta em quantos documentos cada termo aparece
    df = np.zeros(m_termos, dtype=float)
    for tokens in docs_tokenizados:
        for termo_unico in set(tokens):
            idx = termo_para_idx[termo_unico]
            df[idx] += 1.0

   
   
    # aplica formula idf = ln((n + 1) / (df + 1)) + 1
    idf = np.zeros(m_termos, dtype=float)
    for i in range(m_termos):
        idf[i] = math.log((n_docs + 1.0) / (df[i] + 1.0)) + 1.0

    return idf
def construir_matriz_termo_documento(documentos):


    # monta a matriz termo-documento a (m x n) com pesos tf-idf

    docs_tokenizados = tokenizar_colecao(documentos)
    vocabulario, termo_para_idx = construir_vocabulario(docs_tokenizados)
    idf = calcular_idf(docs_tokenizados, vocabulario, termo_para_idx)

    m_termos = len(vocabulario)
    n_docs = len(documentos)
    A = np.zeros((m_termos, n_docs), dtype=float)

    for j, tokens in enumerate(docs_tokenizados):
        total_tokens_doc = len(tokens)
        if total_tokens_doc == 0:
            continue


        # conta frequencia local de cada termo no documento j

        contagem_local = {}
        for termo in tokens:
            contagem_local[termo] = contagem_local.get(termo, 0) + 1

        # preenche a matriz com tf * idf


        for termo, freq in contagem_local.items():
            i = termo_para_idx[termo]
            tf = freq / total_tokens_doc
            A[i, j] = tf * idf[i]

    return A, vocabulario, termo_para_idx, idf




# funcoes de decomposicao svd e reducao dimensional


def calcular_energia_preservada(valores_singulares, k):
    # calcula o percentual de energia mantido pelos k maiores valores singulares
    energia_total = np.sum(valores_singulares ** 2)
    if energia_total == 0.0:
        return 0.0
    energia_retida = np.sum(valores_singulares[:k] ** 2)
    return float((energia_retida / energia_total) * 100.0)


def inverter_matriz_diagonal_segura(valores_singulares_k, tolerancia=1e-10):


    # inverte a diagonal sigma_k ignorando valores proximos de zero


    k = len(valores_singulares_k)
    sigma_inv = np.zeros((k, k), dtype=float)

    for i in range(k):
        valor = valores_singulares_k[i]
        if valor > tolerancia:
            sigma_inv[i, i] = 1.0 / valor
        else:
            sigma_inv[i, i] = 0.0

    return sigma_inv


def executar_svd_e_truncagem(A, k):



    # fatora a matriz a via svd e reduz a dimensionalidade para o posto k


    U, valores_singulares, Vt = np.linalg.svd(A, full_matrices=False)
    Sigma = np.diag(valores_singulares)

    k_efetivo = min(k, len(valores_singulares))
    energia_preservada = calcular_energia_preservada(valores_singulares, k_efetivo)



    # trunca as matrizes para os k primeiros componentes


    U_k = U[:, :k_efetivo]
    sigmas_k = valores_singulares[:k_efetivo]
    Sigma_k = np.diag(sigmas_k)
    Vt_k = Vt[:k_efetivo, :]


    # calcula inversa de sigma_k e projeta documentos no espaco reduzido



    Sigma_k_inv = inverter_matriz_diagonal_segura(sigmas_k)
    docs_espaco_lsi = np.dot(Sigma_k, Vt_k)

    dimensoes_matrizes = {
        "A": A.shape,
        "U": U.shape,
        "Sigma": Sigma.shape,
        "Vt": Vt.shape,
        "U_k": U_k.shape,
        "Sigma_k": Sigma_k.shape,
        "Vt_k": Vt_k.shape,
    }

    return U_k, Sigma_k, Sigma_k_inv, docs_espaco_lsi, energia_preservada, dimensoes_matrizes


# funcoes de consulta, projecao e similaridade


def vetorizar_consulta_usuario(consulta, termo_para_idx, idf):
    # transforma a consulta em um vetor tf-idf de dimensao m
    tokens = normalizar_e_tokenizar(consulta)
    m_termos = len(termo_para_idx)
    q = np.zeros(m_termos, dtype=float)
    tokens_reconhecidos = []
    if len(tokens) == 0:
        return q, tokens_reconhecidos



    # filtra e conta termos da consulta presentes no vocabulario


    contagem = {}
    for termo in tokens:
        if termo in termo_para_idx:
            contagem[termo] = contagem.get(termo, 0) + 1
            if termo not in tokens_reconhecidos:
                tokens_reconhecidos.append(termo)

    # preenche vetor q com tf * idf
    total_tokens = len(tokens)
    for termo, freq in contagem.items():
        i = termo_para_idx[termo]
        tf = freq / total_tokens
        q[i] = tf * idf[i]

    return q, tokens_reconhecidos

def projetar_consulta_espaco_lsi(q, U_k, Sigma_k_inv):

    # projeta o vetor q para o espaco reduzido de dimensao k


    return np.dot(np.dot(q, U_k), Sigma_k_inv)

def similaridade_cosseno(vetor_a, vetor_b):



    # calcula o cosseno do angulo entre dois vetores

    produto_interno = np.dot(vetor_a, vetor_b)
    norma_a = np.linalg.norm(vetor_a)
    norma_b = np.linalg.norm(vetor_b)

    if norma_a == 0.0 or norma_b == 0.0:
        return 0.0

    return float(produto_interno / (norma_a * norma_b))
def pesquisar(consulta, A, U_k, Sigma_k_inv, docs_espaco_lsi, termo_para_idx, idf):


    # calcula scores de similaridade sem svd (tf-idf) e com svd (lsi)

    n_docs = A.shape[1]
    q, tokens_reconhecidos = vetorizar_consulta_usuario(consulta, termo_para_idx, idf)


    # busca tradicional no espaco original  

    scores_sem_svd = []
    for j in range(n_docs):
        score = similaridade_cosseno(q, A[:, j])
        scores_sem_svd.append(score)





    # busca semantica no espaco reduzido


    q_lsi = projetar_consulta_espaco_lsi(q, U_k, Sigma_k_inv)
    scores_com_svd = []
    for j in range(n_docs):
        score = similaridade_cosseno(q_lsi, docs_espaco_lsi[:, j])
        scores_com_svd.append(score)

    return scores_sem_svd, scores_com_svd, tokens_reconhecidos






# funcoes de classificacao e relatorio

def ordenar_indices_por_score(scores):
    # ordena indices dos documentos do maior para o menor score

    indices = list(range(len(scores)))
    return sorted(indices, key=lambda i: scores[i], reverse=True)
def mapear_posicoes_ranking(indices_ordenados):





    # mapeia cada indice de documento a sua posicao no ranking
    mapa_posicoes = {}
    for posicao, doc_idx in enumerate(indices_ordenados):
        mapa_posicoes[doc_idx] = posicao + 1
    return mapa_posicoes


def gerar_relatorio_analitico(consulta, documentos, A, U_k, Sigma_k_inv, docs_espaco_lsi,
                              termo_para_idx, idf, k, energia_preservada, dimensoes_matrizes):
    # monta o texto completo do relatorio comparativo entre os dois metodos




    scores_sem, scores_com, tokens_validos = pesquisar(
        consulta, A, U_k, Sigma_k_inv, docs_espaco_lsi, termo_para_idx, idf
    )

    if len(tokens_validos) == 0:
        return (
            f"\n[AVISO] Nenhum termo da consulta '{consulta}' existe no vocabulário da base.\n"
        )
    rank_sem = ordenar_indices_por_score(scores_sem)
    rank_com = ordenar_indices_por_score(scores_com)
    pos_sem = mapear_posicoes_ranking(rank_sem)
    pos_com = mapear_posicoes_ranking(rank_com)

    linhas = []
    linhas.append("\nRELATÓRIO ANALÍTICO (TF-IDF vs LSI)")
    linhas.append(f"Consulta: \"{consulta}\" | Termos indexados: {', '.join(tokens_validos)}")
    linhas.append(f"Acervo: {A.shape[1]} documentos, {A.shape[0]} termos | k = {k} | Energia preservada: {energia_preservada:.2f}%")
    linhas.append(
        f"Dimensões -> A: {dimensoes_matrizes['A']}, U: {dimensoes_matrizes['U']}, "
        f"Sigma: {dimensoes_matrizes['Sigma']}, V^T: {dimensoes_matrizes['Vt']} | "
        f"U_k: {dimensoes_matrizes['U_k']}, Sigma_k: {dimensoes_matrizes['Sigma_k']}, V^T_k: {dimensoes_matrizes['Vt_k']}"
    )

    linhas.append("\nTop 5 - Busca Tradicional (sem SVD):")
    top_n = min(5, len(documentos))
    for p in range(top_n):
        idx_sem = rank_sem[p]
        linhas.append(f"  {p+1}. Doc {idx_sem+1:02d} (score: {scores_sem[idx_sem]:.4f}) - {documentos[idx_sem]}")

    linhas.append(f"\nTop 5 - Busca com LSI (SVD k={k}):")
    for p in range(top_n):
        idx = rank_com[p]
        variacao = pos_sem[idx] - pos_com[idx]
        if variacao > 0:
            status = f"subiu {variacao} pos."
        elif variacao < 0:
            status = f"desceu {abs(variacao)} pos."
        else:
            status = "manteve pos."

        linhas.append(
            f"  {p+1}. Doc {idx+1:02d} (LSI: {scores_com[idx]:.4f} | TF-IDF: {scores_sem[idx]:.4f} | "
            f"era #{pos_sem[idx]}, {status}) - {documentos[idx]}"
        )

    # identifica documentos sem correspondencia exata recuperados pela lsi
    sem_match_exato = []
    for i in rank_com:
        if scores_sem[i] == 0.0 and scores_com[i] > 0.0:
            sem_match_exato.append(i)

    linhas.append("\nResumo comparativo:")
    linhas.append(
        f"- Com correspondência exata (TF-IDF > 0): "
        f"{sum(1 for s in scores_sem if s > 0.0)} de {len(documentos)} documentos"
    )
    linhas.append(
        f"- Com score positivo na LSI (LSI > 0): "
        f"{sum(1 for s in scores_com if s > 0.0)} de {len(documentos)} documentos"
    )
    linhas.append(
        f"- Recuperados sem correspondência exata (TF-IDF = 0 e LSI > 0): "
        f"{len(sem_match_exato)} documentos"
    )

    if len(sem_match_exato) > 0:
        for i in sem_match_exato:
            linhas.append(f"  * Doc {i+1:02d} (LSI: {scores_com[i]:.4f}) - {documentos[i]}")

    linhas.append(
        "\nObs: A similaridade na LSI vem da projeção no espaço reduzido (coocorrência estatística) "
        "e pode incluir aproximações residuais sem relação semântica real.\n"
    )

    return "\n".join(linhas)

def salvar_relatorio_txt(conteudo_relatorio, caminho_arquivo):
    # salva o relatorio em arquivo txt utf-8
    with open(caminho_arquivo, "w", encoding="utf-8") as arquivo:
        arquivo.write(conteudo_relatorio)








# funcao principal

def main():
    # constroi matriz termo-documento
    A, vocabulario, termo_para_idx, idf = construir_matriz_termo_documento(base_documentos)

    # aplica svd e reducao para k dimensoes

    U_k, Sigma_k, Sigma_k_inv, docs_espaco_lsi, energia_preservada, dimensoes_matrizes = executar_svd_e_truncagem(
        A, k_dimensoes
    )

    print(
        f"\nSistema LSI pronto | Matriz A: {dimensoes_matrizes['A']} | "
        f"k = {k_dimensoes} | Energia preservada: {energia_preservada:.2f}%"
    )




    # loop para consultas decididas pelo usuario

    contador = 0
    while True:
        consulta = input("\nDigite sua consulta por palavras-chave (ou 'sair' para encerrar): ").strip()
        if consulta.lower() == "sair":
            break
        if not consulta:
            continue

        contador += 1
        relatorio = gerar_relatorio_analitico(
            consulta, base_documentos, A, U_k, Sigma_k_inv, docs_espaco_lsi,
            termo_para_idx, idf, k_dimensoes, energia_preservada, dimensoes_matrizes
        )
        print(relatorio)

        if input("Deseja salvar este relatório em .txt? (s/n): ").strip().lower() == "s":
            nome_arquivo = f"relatorio_lsi_{contador}.txt"
            salvar_relatorio_txt(relatorio, nome_arquivo)
            print(f"Relatório salvo em '{nome_arquivo}'.")


if __name__ == "__main__":
    main()