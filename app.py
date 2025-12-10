import streamlit as st
import pandas as pd
import requests
import json
import time
import numpy as np
import concurrent.futures
from datetime import datetime
from typing import List

from sklearn.metrics.pairwise import cosine_similarity
from google.genai import types
from google import genai 

from pydantic import BaseModel, Field


# --- Configuração das Chaves de API ---
# Certifique-se de que st.secrets esteja configurado corretamente
JINA_API_KEY = st.secrets['JINA_API_KEY']
GEMINI_API_KEY = st.secrets['GEMINI_API_KEY']

def buscar_google_news(termo):
    from GoogleNews import GoogleNews

    # Inicializa o objeto GoogleNews com os parâmetros desejados
    googlenews = GoogleNews(
        lang='pt-BR',        # Define o idioma para português do Brasil
        period='1d',         # Define o período para os últimos 7 dias (ou 1d conforme seu código)
        encode='utf-8'       # Define a codificação para UTF-8
    )

    # Realiza a busca por notícias
    googlenews.search(termo)

    # Define o número máximo de resultados desejados
    max_resultados = 1000
    resultados = []
    pagina = 1

    # Itera sobre as páginas de resultados até atingir o número desejado
    while len(resultados) < max_resultados:
        googlenews.get_page(pagina)
        noticias = googlenews.result()
        if not noticias:
            break  # Encerra se não houver mais resultados
        resultados.extend(noticias)
        pagina += 1

    # Limita a lista de resultados ao número máximo desejado
    resultados = resultados[:max_resultados]

    # Separando as noticias
    links_noticias = [noticia['link'].split('&ved')[0] for noticia in resultados]

    # Exibe os resultados
    quantidade_noticias = len(resultados)
    print(f'Quantidade de notícias retornadas do GoogleNews: {quantidade_noticias}')

    # Coloca todas as noticias num dataframe
    df = pd.DataFrame(resultados)
    if not df.empty:
        df['link'] = df['link'].str.split('&ved').str[0]
        # a coluna media deve ser renomeada para source
        if 'media' in df.columns:
            df.rename(columns={'media': 'source'}, inplace=True)
        # dropar datetime e img se existirem
        cols_to_drop = [col for col in ['datetime', 'img'] if col in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)

    return df

def pega_noticias(INTERESSE):
    """
    Usa o interesse do usuário, utiliza o Gemini para extrair palavras-chave e gerar temas de busca,
    pesquisa no Google News para cada tema e retorna um DataFrame combinado e limpo.
    """
    client = genai.Client(api_key = GEMINI_API_KEY)

    prompt = f"""
    Dado o seguinte interesse do usuário, extraia palavras-chave relevantes e gere 3 a 5 temas de busca relacionados que podem ser usados para encontrar notícias no Google News.
    Formato de saída: Uma lista de strings, onde cada string é um tema de busca.
    Interesse do usuário: {INTERESSE}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": {"type": "array", "items": {"type": "string"}},
            },
        )
        search_themes = json.loads(response.text)
        print("Temas de busca gerados por Gemini:", search_themes)
    except Exception as e:
        print(f"Erro ao gerar temas de busca com Gemini: {e}")
        return pd.DataFrame() 

    all_news_df = pd.DataFrame()

    for theme in search_themes:
        print(f"Buscando notícias para o tema: {theme}")
        news_df = buscar_google_news(theme)
        if not news_df.empty:
            news_df['search_theme'] = theme
            all_news_df = pd.concat([all_news_df, news_df], ignore_index=True)

    if not all_news_df.empty:
        all_news_df.dropna(subset=['link'], inplace=True)
        all_news_df = all_news_df.drop_duplicates(subset=['link'], keep='first')
        if 'title' in all_news_df.columns:
            all_news_df = all_news_df.drop_duplicates(subset=['title'], keep='first')
        all_news_df.reset_index(drop=True, inplace=True)

    print(f"Busca combinada concluída! {all_news_df.shape[0]} notícias únicas encontradas.")
    return all_news_df

def ordenar_noticias_por_similaridade(interesse, df_noticias, top_n=10):
    if df_noticias.empty:
        return df_noticias

    df_noticias = df_noticias[df_noticias['title'].notna()]  # Remove NaN
    df_noticias = df_noticias[df_noticias['title'].str.strip() != ""] # Remove vazios
    df_noticias.reset_index(drop=True, inplace=True) # Reseta o índice para alinhar com o futuro loop

    if df_noticias.empty:
        return df_noticias

    TEXTOS = df_noticias['title'].to_list()

    client = genai.Client(api_key = st.secrets['GEMINI_API_KEY'])

    # Gera o embedding do interesse do usuário
    try:
        result = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=interesse)
        interesse_embed = np.array(result.embeddings[0].values)
    except Exception as e:
        print(f"Erro ao gerar embedding do interesse: {e}")
        return df_noticias.head(top_n)

    VETORES = []
    
    for i in range(0, len(TEXTOS), 100):
        batch_textos = TEXTOS[i:i+100]
        try:
            embeddings_result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=batch_textos,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            ).embeddings
            VETORES.extend([np.array(e.values) for e in embeddings_result])
        except Exception as e:
            print(f"Erro ao gerar embeddings do lote {i}: {e}")
            
            dimensao = len(interesse_embed)
            VETORES.extend([np.zeros(dimensao) for _ in batch_textos])

    if len(VETORES) != len(df_noticias):
        print(f"Aviso: Discrepância de tamanho. Dataframe: {len(df_noticias)}, Vetores: {len(VETORES)}")
        # Ajuste de segurança: corta o maior para igualar o menor
        min_len = min(len(df_noticias), len(VETORES))
        df_noticias = df_noticias.iloc[:min_len]
        VETORES = VETORES[:min_len]

    interesse_embed_2d = interesse_embed.reshape(1, -1)
    
    if VETORES:
        similaridades = [cosine_similarity(interesse_embed_2d, v.reshape(1, -1))[0][0] for v in VETORES]
        df_noticias['score'] = similaridades
        df_noticias.sort_values(by='score', ascending=False, inplace = True)

    return df_noticias.head(top_n).reset_index(drop=True)

def _fetch_single_article(url, headers, title):
    """Função auxiliar para buscar um único artigo (usada no paralelismo)."""
    try:
        response = requests.get(url, headers=headers, timeout=90)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Erro ao buscar conteúdo para o título '{title}': {e}"

def extrair_conteudo_noticias(df_noticias):
    """Extrai o conteúdo completo dos artigos usando a Jina AI API com paralelismo."""
    headers = {
        "Authorization": f"Bearer {st.secrets['JINA_API_KEY']}",
        "X-Engine": "browser",
        "X-Return-Format": "markdown"
    }

    # Dicionário para armazenar resultados indexados pelo índice original do DataFrame
    results = {}
    
    print(f"Iniciando extração paralela de {len(df_noticias)} notícias...")

    # MELHORIA: Paralelismo com ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Mapeia futuros para os índices originais
        future_to_index = {}
        for index, row in df_noticias.iterrows():
            url = f"https://r.jina.ai/{row['link']}"
            future = executor.submit(_fetch_single_article, url, headers, row['title'])
            future_to_index[future] = index

        # Processa conforme as tarefas são concluídas
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                content = future.result()
                results[idx] = content
            except Exception as e:
                results[idx] = f"Erro inesperado na thread: {e}"

    # Atribui os resultados de volta ao DataFrame garantindo a ordem correta
    # Cria uma lista de conteúdos ordenada pelo índice do DataFrame
    conteudos_ordenados = [results[i] for i in df_noticias.index]
    df_noticias['content'] = conteudos_ordenados
    
    return df_noticias

def processa_noticias_com_gemini(articles_df):
    client = genai.Client(api_key = GEMINI_API_KEY)

    class Noticia(BaseModel):
        titulo: str = Field(..., description="O título da notícia.")
        data_de_publicacao: str = Field(..., description="A data em que a notícia foi publicada. Use sempre o formato: 'DD/MM/AAAA'.")
        autor: str = Field(..., description="O nome do autor da notícia.")
        portal: str = Field(..., description="O nome do portal de notícias onde a notícia foi publicada.")
        resumo_curto: str = Field(..., description="Um resumo conciso da notícia em torno de 50 palavras. De preferência para colocar informação adicional ao titulo (nao repetir a informacao do titulo)")
        resumo_maior: str = Field(..., description="Um resumo mais detalhado da notícia em torno de 500 palavras.")
        pontos_principais: List[str] = Field(..., description="um resumo da noticia em formato de lista item a item")
        noticia_completa: str = Field(..., description="O texto completo da notícia.")
        links_de_imagens: List[str] = Field(..., description="Uma lista de URLs das imagens associadas à notícia. Considere apenas aquelas relevantes para a noticia. Descarte logos, divulgacoes, etc...")
        tags_relevantes: List[str] = Field(..., description="Uma lista de tags ou palavras-chave relevantes para a notícia.")
        prompt_satira_imagem: str = Field(..., description="Um prompt de sátira, baseado no conteúdo da notícia, para ser usado em um gerador de imagens. Deve ser criativo e com um tom humorístico ou irônico.")


    respostas = []
    for texto in articles_df['content']:
        # MELHORIA: Feedback de Falha na Extração (Evita chamar a IA para erros)
        if "Erro ao buscar conteúdo" in texto or "Erro inesperado" in texto:
            print("Conteúdo com erro detectado. Pulando processamento IA e inserindo placeholder.")
            # Cria um JSON dummy para manter o alinhamento
            dummy_data = {
                "titulo": "Erro ao acessar notícia",
                "data_de_publicacao": datetime.now().strftime("%d/%m/%Y"),
                "autor": "Sistema",
                "portal": "N/A",
                "resumo_curto": "Não foi possível extrair o conteúdo desta notícia.",
                "resumo_maior": "O sistema de extração não conseguiu recuperar o texto deste link.",
                "pontos_principais": [],
                "noticia_completa": "",
                "links_de_imagens": [],
                "tags_relevantes": ["erro"],
                "prompt_satira_imagem": ""
            }
            respostas.append(json.dumps(dummy_data))
            continue

        print(f"Fazendo extração do {texto[:40]}...")
        while True:
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents = f"Extraia informacoes da noticia em texto cru dada a seguir: \n\n {texto}",
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": Noticia,
                    },
                )
                respostas.append(response.text)
                break
            except Exception as e:
                print(f"Erro na API: {e} \nTentando novamente em 3s...")
                time.sleep(3)

    # Converter cada string JSON em um dicionário
    lista_de_dicionarios = [json.loads(json_string) for json_string in respostas]

    processados_df = pd.DataFrame(lista_de_dicionarios)
    return processados_df


Este código cobre a parte de geração de HTML e a interface do Streamlit. Fiz várias melhorias focadas na experiência do usuário e no funcionamento correto dentro do ambiente do Streamlit.

Principais Melhorias Realizadas:
Persistência com st.session_state: O maior problema de apps Streamlit é que, ao clicar em um botão, o app roda de novo e perde o que estava na tela. Adicionei lógica para salvar a newsletter gerada na memória, assim ela não some.

Botão de Download: Adicionei um botão nativo (st.download_button) para o usuário baixar o arquivo .html final.

Correção do CSS (Pop-up): Mudei a posição dos modais (popovers) de absolute para fixed.

Por que? Com absolute, se o usuário rolar a página até a última notícia e clicar em "Resumo", o pop-up abriria lá no topo da página, fora de visão. Com fixed, ele abre sempre no meio da tela visível.

Preview Nativo: Adicionei uma aba "Dados Brutos" para você conseguir inspecionar a tabela caso o HTML falhe ou fique desformatado.

Aqui está o código melhorado. Substitua a parte final do seu arquivo (as funções gerar_card..., gerar_html... e a parte da Interface Streamlit) por este bloco:

Python

def gerar_card_noticia(noticia: dict, idx: int) -> str:
    """Gera HTML para um card de notícia a partir de um dicionário."""

    titulo = noticia.get('titulo', 'Sem título')
    portal = noticia.get('portal', '')
    data_pub = noticia.get('data_de_publicacao', '')
    resumo_breve = noticia.get('resumo_curto', '')
    resumo_expandido = noticia.get('resumo_maior', '')
    tags = noticia.get('tags_relevantes', [])
    # Garante que link existe; usa # se não tiver
    url_original = noticia.get('link', '#') 
    caminho_imagem = noticia.get('links_de_imagens', [])
    prompt_satira_imagem = noticia.get('prompt_satira_imagem', '')
    pontos_principais = noticia.get('pontos_principais', [])

    # Tratamento seguro para listas vazias ou None
    imagem_url = caminho_imagem[0] if (isinstance(caminho_imagem, list) and caminho_imagem) else ''
    tags_str = ', '.join(tags) if (isinstance(tags, list) and tags) else ''
    
    pontos_principais_html = ""
    if isinstance(pontos_principais, list) and pontos_principais:
        pontos_principais_html = "".join([f"<li>{p}</li>" for p in pontos_principais])

    card_html = f"""
    <div class="card-noticia">
        <div class="card-header">
            <h3 class="card-titulo">{titulo}</h3>
            <div class="card-meta">
                <span class="portal">{portal}</span>
                <span class="data">{data_pub}</span>
            </div>
        </div>

        <div class="card-content">
            <div class="card-imagem">
                <img src="{imagem_url}" alt="{titulo}" onerror="this.style.display='none'">
            </div>

            <div class="card-texto">
                <p class="resumo-breve"><strong>Resumo Curto:</strong> {resumo_breve}</p>

                <div class="button-array">
                    <button class="popover-button" data-popover-target="#popover-resumo-{idx}">Resumo Completo</button>
                    <button class="popover-button" data-popover-target="#popover-tags-{idx}">Tags</button>
                    <button class="popover-button" data-popover-target="#popover-satira-{idx}">Prompt Imagem</button>
                    <button class="popover-button" data-popover-target="#popover-pontos-{idx}">Pontos Chave</button>
                </div>

                <a href="{url_original}" target="_blank" class="btn-ler-mais">
                    Ler notícia completa
                </a>
            </div>
        </div>

        <div id="popover-resumo-{idx}" class="popover-content">
            <div class="popover-header">
                <h4>Resumo Completo</h4>
                 <span class="close-popover">&times;</span>
            </div>
            <div class="popover-body">
                <p>{resumo_expandido}</p>
            </div>
        </div>

        <div id="popover-tags-{idx}" class="popover-content">
             <div class="popover-header">
                <h4>Tags Relevantes</h4>
                 <span class="close-popover">&times;</span>
            </div>
             <div class="popover-body">
                 <div>{tags_str}</div>
            </div>
        </div>

        <div id="popover-satira-{idx}" class="popover-content">
            <div class="popover-header">
                <h4>Prompt de Sátira</h4>
                <span class="close-popover">&times;</span>
            </div>
             <div class="popover-body">
                 <p>{prompt_satira_imagem}</p>
            </div>
        </div>

        <div id="popover-pontos-{idx}" class="popover-content">
            <div class="popover-header">
                <h4>Pontos Principais</h4>
                 <span class="close-popover">&times;</span>
            </div>
             <div class="popover-body">
                 <ul>{pontos_principais_html}</ul>
            </div>
        </div>
    </div>
    """
    return card_html

def gerar_html_newsletter(df: pd.DataFrame, interesse: str) -> str:
    """Gera um arquivo HTML para a newsletter a partir do DataFrame processado."""

    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Newsletter: {interesse}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background-color: #f0f2f5; color: #333; }} 
            .container {{ max-width: 800px; margin: auto; background: #fff; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); }}
            .header {{ text-align: center; margin-bottom: 40px; border-bottom: 2px solid #eee; padding-bottom: 20px; }}
            .header h1 {{ color: #2c3e50; margin: 0; font-size: 2.5em; }}
            
            /* Card Styles */
            .card-noticia {{ border: 1px solid #e1e4e8; margin-bottom: 30px; border-radius: 12px; overflow: hidden; background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.05); transition: transform 0.2s; }} 
            .card-noticia:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
            
            .card-header {{ background: #f8f9fa; padding: 15px 20px; border-bottom: 1px solid #e1e4e8; }} 
            .card-titulo {{ margin: 0; color: #1a73e8; font-size: 1.25em; }}
            .card-meta {{ font-size: 0.85em; color: #666; margin-top: 8px; display: flex; gap: 15px; }}
            
            .card-content {{ display: flex; flex-wrap: wrap; padding: 20px; gap: 20px; }}
            .card-imagem {{ flex: 1 1 200px; max-width: 300px; }}
            .card-imagem img {{ width: 100%; height: auto; border-radius: 8px; object-fit: cover; }}
            .card-texto {{ flex: 2 1 300px; display: flex; flex-direction: column; }}
            
            .resumo-breve {{ font-size: 1em; color: #444; margin-bottom: 20px; }}
            
            /* Buttons */
            .button-array {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }} 
            .popover-button {{ flex: 1; min-width: 120px; padding: 8px 12px; background-color: #e8f0fe; color: #1967d2; border: 1px solid #dadce0; border-radius: 4px; cursor: pointer; font-size: 0.9em; font-weight: 500; transition: background 0.2s; }}
            .popover-button:hover {{ background-color: #d2e3fc; }}
            
            .btn-ler-mais {{ margin-top: auto; display: block; width: 100%; background-color: #1a73e8; color: white; padding: 12px; text-decoration: none; border-radius: 6px; text-align: center; font-weight: bold; transition: background 0.2s; }}
            .btn-ler-mais:hover {{ background-color: #1557b0; }}

            /* Popovers (Modais) FIXO PARA FUNCIONAR NO STREAMLIT */
            .popover-content {{ 
                display: none; 
                position: fixed; /* IMPORTANTE: Fixed para centralizar na tela visível */
                top: 50%; 
                left: 50%; 
                transform: translate(-50%, -50%); 
                width: 90%;
                max-width: 500px;
                max-height: 80vh;
                overflow-y: auto;
                background-color: #fff; 
                box-shadow: 0 10px 25px rgba(0,0,0,0.2); 
                padding: 20px; 
                z-index: 9999; 
                border-radius: 12px; 
                border: 1px solid #ddd; 
            }}
            .popover-header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 15px; }}
            .popover-header h4 {{ margin: 0; color: #333; }}
            .close-popover {{ color: #999; font-size: 24px; font-weight: bold; cursor: pointer; line-height: 1; }}
            .close-popover:hover {{ color: #333; }}
            
            /* Overlay Fundo Escuro */
            .overlay {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 9998; }}
        </style>
    </head>
    <body>
        <div class="overlay" id="modal-overlay"></div>
        <div class="container">
            <div class="header">
                <h1>Newsletter: {interesse}</h1>
                <p>Gerada automaticamente por IA</p>
            </div>
            """

    for idx, row in df.iterrows():
        noticia_dict = row.to_dict()
        html_content += gerar_card_noticia(noticia_dict, idx)

    html_content += """
        </div>
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const overlay = document.getElementById('modal-overlay');
                
                function fecharTodos() {
                    document.querySelectorAll('.popover-content').forEach(el => el.style.display = 'none');
                    overlay.style.display = 'none';
                }

                // Abrir popover
                document.querySelectorAll('.popover-button').forEach(button => {
                    button.addEventListener('click', function(e) {
                        e.stopPropagation();
                        const targetId = this.dataset.popoverTarget;
                        const target = document.querySelector(targetId);
                        
                        fecharTodos(); // Fecha outros abertos
                        
                        if(target) {
                            target.style.display = 'block';
                            overlay.style.display = 'block';
                        }
                    });
                });

                // Fechar ao clicar no X
                document.querySelectorAll('.close-popover').forEach(btn => {
                    btn.addEventListener('click', fecharTodos);
                });

                // Fechar ao clicar no fundo escuro
                overlay.addEventListener('click', fecharTodos);
            });
        </script>
    </body>
    </html>
    """
    return html_content

# --- Interface Streamlit ---

st.title('🤖 Minha Newsletter com IA')

# Sidebar para inputs deixa a interface principal mais limpa
with st.sidebar:
    st.header("Configurações")
    INTERESSE = st.text_input('Tema de Interesse', placeholder="Ex: IA na Medicina")
    TOP_NOTICIAS = st.number_input('Número de Notícias', min_value=1, max_value=20, value=5)
    btn_gerar = st.button('Gerar Newsletter', type="primary")

# Inicializa Session State para guardar os dados se ainda não existirem
if "newsletter_data" not in st.session_state:
    st.session_state.newsletter_data = None
if "newsletter_html" not in st.session_state:
    st.session_state.newsletter_html = None

# Lógica de Geração
if btn_gerar:
    if not INTERESSE:
        st.warning('⚠️ Por favor, preencha o seu Interesse específico na barra lateral.')
    else:
        try:
            # 1. Pega as notícias
            with st.spinner('📰 Buscando as notícias mais recentes...'):
                pegas = pega_noticias(INTERESSE)
            
            if pegas.empty:
                st.error("Nenhuma notícia encontrada para este tema. Tente termos mais gerais.")
            else:
                # 2. Ordena
                with st.spinner('🧠 Analisando relevância com Gemini...'):
                    # (Certifique-se que sua função 'ordenar...' tem a correção de strings vazias que passei antes)
                    top_noticias = ordenar_noticias_por_similaridade(
                        interesse=INTERESSE,
                        df_noticias=pegas,
                        top_n=int(TOP_NOTICIAS) 
                    )

                # 3. Extrai conteúdo
                with st.spinner('📄 Lendo o conteúdo das notícias (Jina AI)...'):
                    extracoes = extrair_conteudo_noticias(top_noticias)

                # 4. Processa com IA
                with st.spinner('✨ Gerando resumos e insights...'):
                    processados = processa_noticias_com_gemini(extracoes)

                # 5. Montagem Final (Garante alinhamento)
                with st.spinner('🔨 Montando HTML...'):
                    # Reseta índices para garantir concatenação correta
                    extracoes.reset_index(drop=True, inplace=True)
                    processados.reset_index(drop=True, inplace=True)
                    
                    final_df = pd.concat([extracoes, processados], axis=1)
                    
                    # Gera o HTML e Salva no Session State
                    st.session_state.newsletter_data = final_df
                    st.session_state.newsletter_html = gerar_html_newsletter(final_df, INTERESSE)
                    
                    st.success('Newsletter gerada com sucesso!')

        except Exception as e:
            st.error(f"Ocorreu um erro durante o processamento: {e}")
            # Dica: imprima o erro completo no terminal para debug
            print(e)

# Exibição dos Resultados (Se existirem no Session State)
if st.session_state.newsletter_html:
    
    st.divider()
    
    # Colunas para botões de ação
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Botão de Download do HTML
        st.download_button(
            label="📥 Baixar Newsletter HTML",
            data=st.session_state.newsletter_html,
            file_name=f"newsletter_{INTERESSE.replace(' ', '_')}.html",
            mime="text/html"
        )
    
    with col2:
        if st.button("🔄 Limpar Resultados"):
            st.session_state.newsletter_data = None
            st.session_state.newsletter_html = None
            st.rerun()

    # Abas para visualização
    tab1, tab2 = st.tabs(["📧 Visualização Web", "📊 Dados Brutos"])
    
    with tab1:
        st.caption("Esta é uma prévia interativa. Use o botão de download para ver em tela cheia no seu navegador.")
        # Height aumentado e scrolling ativado
        st.components.v1.html(st.session_state.newsletter_html, height=800, scrolling=True)
        
    with tab2:
        st.caption("Tabela com os dados extraídos e gerados pela IA.")
        st.dataframe(st.session_state.newsletter_data)
