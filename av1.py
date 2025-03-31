import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Configurações para melhorar a visualização
plt.rcParams['figure.figsize'] = (12, 8)
sns.set(style="whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Definir o caminho para o arquivo de microdados
# Atualize este caminho para o local onde você salvou os arquivos
arquivo_dados = "MICRODADOS_ENEM_2023.csv"

# Verificar se o arquivo existe
if not os.path.exists(arquivo_dados):
    print(f"Arquivo não encontrado: {arquivo_dados}")
    print("Por favor, verifique o caminho e se você baixou os microdados do ENEM 2023.")
else:
    print(f"Arquivo encontrado: {arquivo_dados}")
    print("Iniciando carregamento dos dados...")

# Como o arquivo de microdados é muito grande, vamos carregar apenas as colunas necessárias
# Definir as colunas que precisamos para nossa análise
colunas_necessarias = [
    # Identificação e características do participante
    'NU_INSCRICAO', 'TP_SEXO', 'TP_FAIXA_ETARIA', 'TP_ESTADO_CIVIL', 
    'TP_COR_RACA', 'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO', 
    
    # Escola
    'TP_ESCOLA', 'TP_ENSINO', 'CO_MUNICIPIO_ESC', 'CO_UF_ESC', 'TP_LOCALIZACAO_ESC',
    
    # Socioeconômico
    'Q001', 'Q002', 'Q003', 'Q004', 'Q005', 'Q006', 'Q007', 'Q008', 'Q009', 'Q010',
    'Q011', 'Q012', 'Q013', 'Q014', 'Q015', 'Q016', 'Q017', 'Q018', 'Q019', 'Q020',
    'Q021', 'Q022', 'Q023', 'Q024', 'Q025',
    
    # Notas
    'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO',
    
    # Presença
    'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT',
    
    # Local de prova
    'CO_MUNICIPIO_PROVA', 'CO_UF_PROVA'
]

try:
    # Carregar apenas as colunas necessárias para economizar memória
    print("Carregando dados... (isso pode levar alguns minutos)")
    # Usamos o parâmetro low_memory=False para evitar warnings com tipos mistos nas colunas
    dados = pd.read_csv(arquivo_dados, sep=';', encoding='latin-1', 
                         usecols=colunas_necessarias, low_memory=False)
    
    print(f"Dados carregados com sucesso! Formato: {dados.shape}")
    
    # Mostrar informações básicas do DataFrame
    print("\nInformações dos dados:")
    print(dados.info())
    
    # Mostrar primeiras linhas para verificar os dados
    print("\nPrimeiras linhas dos dados:")
    print(dados.head())
    
except Exception as e:
    print(f"Erro ao carregar os dados: {e}")
    print("Verifique se o arquivo está no formato correto e se o caminho está correto.")

# Função para limpar e preparar os dados
def preparar_dados(df):
    """
    Limpa e prepara os dados para análise, incluindo:
    - Remoção de valores ausentes nas colunas relevantes
    - Tratamento de outliers
    - Criação de features adicionais
    
    Args:
        df: DataFrame com os dados originais
    
    Returns:
        DataFrame limpo e preparado para análise
    """
    # Criar uma cópia para não modificar o original
    df_clean = df.copy()
    
    # Filtrar apenas participantes que fizeram a prova de Ciências Humanas
    df_clean = df_clean[df_clean['TP_PRESENCA_CH'] == 1]
    
    # Remover valores ausentes nas notas de CH
    df_clean = df_clean.dropna(subset=['NU_NOTA_CH'])
    
    # Ajustar tipos de dados, se necessário
    # Converter colunas numéricas para o tipo correto
    colunas_numericas = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']
    for col in colunas_numericas:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Criar variáveis derivadas úteis para a análise
    
    # Mapear faixa etária para descrições mais claras
    mapa_faixa_etaria = {
        1: 'Menor de 17 anos',
        2: '17 anos',
        3: '18 anos',
        4: '19 anos',
        5: '20 anos',
        6: '21 anos',
        7: '22 anos',
        8: '23 anos',
        9: '24 anos',
        10: '25 anos',
        11: 'Entre 26 e 30 anos',
        12: 'Entre 31 e 35 anos',
        13: 'Entre 36 e 40 anos',
        14: 'Entre 41 e 45 anos',
        15: 'Entre 46 e 50 anos',
        16: 'Entre 51 e 55 anos',
        17: 'Entre 56 e 60 anos',
        18: 'Entre 61 e 65 anos',
        19: 'Entre 66 e 70 anos',
        20: 'Maior de 70 anos'
    }
    df_clean['FAIXA_ETARIA_DESC'] = df_clean['TP_FAIXA_ETARIA'].map(mapa_faixa_etaria)
    
    # Mapear tipo de escola
    mapa_tipo_escola = {
        1: 'Não Respondeu',
        2: 'Pública',
        3: 'Privada',
        4: 'Exterior'
    }
    df_clean['TIPO_ESCOLA_DESC'] = df_clean['TP_ESCOLA'].map(mapa_tipo_escola)
    
    # Mapear localização da escola
    mapa_localizacao = {
        1: 'Urbana',
        2: 'Rural'
    }
    df_clean['LOCALIZACAO_ESCOLA_DESC'] = df_clean['TP_LOCALIZACAO_ESC'].map(mapa_localizacao)
    
    # Mapear escolaridade dos pais (Q001 e Q002)
    mapa_escolaridade = {
        'A': 'Nunca estudou',
        'B': 'Ensino Fundamental I Incompleto',
        'C': 'Ensino Fundamental I Completo',
        'D': 'Ensino Fundamental II Incompleto',
        'E': 'Ensino Fundamental II Completo',
        'F': 'Ensino Médio Incompleto',
        'G': 'Ensino Médio Completo',
        'H': 'Ensino Superior Incompleto',
        'I': 'Ensino Superior Completo',
        'J': 'Pós-graduação',
    }
    df_clean['ESCOLARIDADE_PAI'] = df_clean['Q001'].map(mapa_escolaridade)
    df_clean['ESCOLARIDADE_MAE'] = df_clean['Q002'].map(mapa_escolaridade)
    
    # Mapear renda familiar (Q006)
    mapa_renda = {
        'A': 'Nenhuma renda',
        'B': 'Até R$ 1.412,00',
        'C': 'R$ 1.412,01 a R$ 2.824,00',
        'D': 'R$ 2.824,01 a R$ 4.236,00',
        'E': 'R$ 4.236,01 a R$ 5.648,00',
        'F': 'R$ 5.648,01 a R$ 7.060,00',
        'G': 'R$ 7.060,01 a R$ 8.472,00',
        'H': 'R$ 8.472,01 a R$ 9.884,00',
        'I': 'R$ 9.884,01 a R$ 11.296,00',
        'J': 'R$ 11.296,01 a R$ 12.708,00',
        'K': 'R$ 12.708,01 a R$ 14.120,00',
        'L': 'R$ 14.120,01 a R$ 15.532,00',
        'M': 'R$ 15.532,01 a R$ 16.944,00',
        'N': 'R$ 16.944,01 a R$ 18.356,00',
        'O': 'R$ 18.356,01 a R$ 19.768,00',
        'P': 'R$ 19.768,01 a R$ 21.180,00',
        'Q': 'Acima de R$ 21.180,01',
    }
    df_clean['RENDA_FAMILIAR'] = df_clean['Q006'].map(mapa_renda)
    
    # Mapear acesso à internet (Q025)
    mapa_internet = {
        'A': 'Sim',
        'B': 'Não'
    }
    df_clean['ACESSO_INTERNET'] = df_clean['Q025'].map(mapa_internet)
    
    # Agrupar renda em categorias mais amplas para facilitar a análise
    df_clean['FAIXA_RENDA'] = pd.cut(
        df_clean['Q006'].astype('str').map(lambda x: 'A B C D E F G H I J K L M N O P Q'.index(x) if x in 'ABCDEFGHIJKLMNOPQ' else -1),
        bins=[-1, 0, 3, 6, 10, 16],  # Agrupamentos da renda
        labels=['Sem resposta', 'Baixa', 'Média-Baixa', 'Média', 'Média-Alta', 'Alta']
    )
    
    return df_clean

# Função para análise da distribuição de notas
def analisar_distribuicao_notas(df):
    """
    Analisa a distribuição de notas da prova de Ciências Humanas.
    
    Args:
        df: DataFrame com os dados preparados
    """
    print("\n--- ANÁLISE DA DISTRIBUIÇÃO DE NOTAS ---")
    
    # Estatísticas descritivas das notas
    print("\nEstatísticas descritivas das notas de Ciências Humanas:")
    print(df['NU_NOTA_CH'].describe())
    
    # Plotar histograma das notas
    plt.figure(figsize=(12, 6))
    sns.histplot(df['NU_NOTA_CH'], kde=True, bins=30)
    plt.title('Distribuição das Notas de Ciências Humanas - ENEM 2023')
    plt.xlabel('Nota')
    plt.ylabel('Frequência')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('distribuicao_notas_ch.png')
    plt.close()
    
    # Boxplot das notas
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df['NU_NOTA_CH'])
    plt.title('Boxplot das Notas de Ciências Humanas - ENEM 2023')
    plt.ylabel('Nota')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('boxplot_notas_ch.png')
    plt.close()
    
    print("\nAnálise de valores atípicos (outliers):")
    q1 = df['NU_NOTA_CH'].quantile(0.25)
    q3 = df['NU_NOTA_CH'].quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    
    print(f"Q1 (25%): {q1:.2f}")
    print(f"Q3 (75%): {q3:.2f}")
    print(f"IQR: {iqr:.2f}")
    print(f"Limite inferior para outliers: {limite_inferior:.2f}")
    print(f"Limite superior para outliers: {limite_superior:.2f}")
    
    outliers = df[(df['NU_NOTA_CH'] < limite_inferior) | (df['NU_NOTA_CH'] > limite_superior)]
    print(f"Número de outliers: {len(outliers)}")
    print(f"Percentual de outliers: {len(outliers) / len(df) * 100:.2f}%")

# Função para analisar a relação entre idade e notas
def analisar_idade_notas(df):
    """
    Analisa a relação entre idade (faixa etária) e notas de Ciências Humanas.
    
    Args:
        df: DataFrame com os dados preparados
    """
    print("\n--- ANÁLISE DA RELAÇÃO ENTRE IDADE E NOTAS ---")
    
    # Estatísticas por faixa etária
    stats_por_idade = df.groupby('FAIXA_ETARIA_DESC')['NU_NOTA_CH'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    stats_por_idade = stats_por_idade.sort_values(by='mean', ascending=False)
    
    print("\nEstatísticas de notas por faixa etária:")
    print(stats_por_idade)
    
    # Plotar boxplot de notas por faixa etária
    plt.figure(figsize=(15, 10))
    # Usar apenas as 12 primeiras faixas etárias para melhor visualização
    faixas_principais = df['FAIXA_ETARIA_DESC'].value_counts().nlargest(12).index
    df_filtrado = df[df['FAIXA_ETARIA_DESC'].isin(faixas_principais)]
    
    # Ordenar por idade aproximada
    ordem_faixas = [f for f in mapa_faixa_etaria.values() if f in faixas_principais]
    
    sns.boxplot(x='FAIXA_ETARIA_DESC', y='NU_NOTA_CH', data=df_filtrado, order=ordem_faixas)
    plt.title('Notas de Ciências Humanas por Faixa Etária - ENEM 2023')
    plt.xlabel('Faixa Etária')
    plt.ylabel('Nota')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('notas_por_faixa_etaria.png')
    plt.close()
    
    # Gráfico de barras para média por faixa etária
    plt.figure(figsize=(15, 8))
    media_por_idade = df.groupby('FAIXA_ETARIA_DESC')['NU_NOTA_CH'].mean().reset_index()
    media_por_idade = media_por_idade.sort_values('FAIXA_ETARIA_DESC', 
                                                 key=lambda x: x.map({v: i for i, v in enumerate(ordem_faixas)}))
    
    sns.barplot(x='FAIXA_ETARIA_DESC', y='NU_NOTA_CH', data=media_por_idade)
    plt.title('Média das Notas de Ciências Humanas por Faixa Etária - ENEM 2023')
    plt.xlabel('Faixa Etária')
    plt.ylabel('Média da Nota')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('media_notas_por_faixa_etaria.png')
    plt.close()

# Função para analisar a relação entre renda e notas
def analisar_renda_notas(df):
    """
    Analisa a relação entre renda familiar e notas de Ciências Humanas.
    
    Args:
        df: DataFrame com os dados preparados
    """
    print("\n--- ANÁLISE DA RELAÇÃO ENTRE RENDA E NOTAS ---")
    
    # Estatísticas por faixa de renda
    stats_por_renda = df.groupby('RENDA_FAMILIAR')['NU_NOTA_CH'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    stats_por_renda = stats_por_renda.sort_values(by='mean', ascending=False)
    
    print("\nEstatísticas de notas por faixa de renda:")
    print(stats_por_renda)
    
    # Plotar boxplot de notas por renda familiar
    plt.figure(figsize=(16, 10))
    
    # Ordenar as categorias de renda do menor para o maior valor
    ordem_renda = [
        'Nenhuma renda',
        'Até R$ 1.412,00',
        'R$ 1.412,01 a R$ 2.824,00',
        'R$ 2.824,01 a R$ 4.236,00',
        'R$ 4.236,01 a R$ 5.648,00',
        'R$ 5.648,01 a R$ 7.060,00',
        'R$ 7.060,01 a R$ 8.472,00',
        'R$ 8.472,01 a R$ 9.884,00',
        'R$ 9.884,01 a R$ 11.296,00',
        'R$ 11.296,01 a R$ 12.708,00',
        'R$ 12.708,01 a R$ 14.120,00',
        'R$ 14.120,01 a R$ 15.532,00',
        'R$ 15.532,01 a R$ 16.944,00',
        'R$ 16.944,01 a R$ 18.356,00',
        'R$ 18.356,01 a R$ 19.768,00',
        'R$ 19.768,01 a R$ 21.180,00',
        'Acima de R$ 21.180,01'
    ]
    
    # Filtrar para incluir apenas as categorias presentes nos dados
    ordem_renda_presente = [r for r in ordem_renda if r in df['RENDA_FAMILIAR'].unique()]
    
    sns.boxplot(x='RENDA_FAMILIAR', y='NU_NOTA_CH', data=df, order=ordem_renda_presente)
    plt.title('Notas de Ciências Humanas por Renda Familiar - ENEM 2023')
    plt.xlabel('Renda Familiar')
    plt.ylabel('Nota')
    plt.xticks(rotation=90)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('notas_por_renda.png')
    plt.close()
    
    # Gráfico de barras para média por renda
    plt.figure(figsize=(16, 8))
    media_por_renda = df.groupby('RENDA_FAMILIAR')['NU_NOTA_CH'].mean().reset_index()
    # Ordenar pela ordem de renda definida
    media_por_renda['ordem'] = media_por_renda['RENDA_FAMILIAR'].map({v: i for i, v in enumerate(ordem_renda)})
    media_por_renda = media_por_renda.sort_values('ordem')
    
    sns.barplot(x='RENDA_FAMILIAR', y='NU_NOTA_CH', data=media_por_renda)
    plt.title('Média das Notas de Ciências Humanas por Renda Familiar - ENEM 2023')
    plt.xlabel('Renda Familiar')
    plt.ylabel('Média da Nota')
    plt.xticks(rotation=90)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('media_notas_por_renda.png')
    plt.close()
    
    # Calcular correlação entre renda e nota (usando o índice numérico da renda)
    df['INDICE_RENDA'] = df['Q006'].astype('str').map(lambda x: 'A B C D E F G H I J K L M N O P Q'.index(x) if x in 'ABCDEFGHIJKLMNOPQ' else np.nan)
    correlacao = df['INDICE_RENDA'].corr(df['NU_NOTA_CH'])
    print(f"\nCorrelação entre índice de renda e nota CH: {correlacao:.4f}")
    
    # Gráfico de dispersão
    plt.figure(figsize=(10, 6))
    sns.regplot(x='INDICE_RENDA', y='NU_NOTA_CH', data=df, scatter_kws={'alpha':0.1}, line_kws={'color':'red'})
    plt.title('Relação entre Renda Familiar e Notas de Ciências Humanas - ENEM 2023')
    plt.xlabel('Índice de Renda (maior valor = maior renda)')
    plt.ylabel('Nota de Ciências Humanas')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('dispersao_renda_notas.png')
    plt.close()

# Função para analisar a relação entre escolaridade dos pais e notas
def analisar_escolaridade_pais_notas(df):
    """
    Analisa a relação entre escolaridade dos pais e notas de Ciências Humanas.
    
    Args:
        df: DataFrame com os dados preparados
    """
    print("\n--- ANÁLISE DA RELAÇÃO ENTRE ESCOLARIDADE DOS PAIS E NOTAS ---")
    
    # Estatísticas por escolaridade do pai
    stats_por_escolaridade_pai = df.groupby('ESCOLARIDADE_PAI')['NU_NOTA_CH'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    stats_por_escolaridade_pai = stats_por_escolaridade_pai.sort_values(by='mean', ascending=False)
    
    print("\nEstatísticas de notas por escolaridade do pai:")
    print(stats_por_escolaridade_pai)
    
    # Estatísticas por escolaridade da mãe
    stats_por_escolaridade_mae = df.groupby('ESCOLARIDADE_MAE')['NU_NOTA_CH'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    stats_por_escolaridade_mae = stats_por_escolaridade_mae.sort_values(by='mean', ascending=False)
    
    print("\nEstatísticas de notas por escolaridade da mãe:")
    print(stats_por_escolaridade_mae)
    
    # Ordem de escolaridade do menor para o maior nível
    ordem_escolaridade = [
        'Nunca estudou',
        'Ensino Fundamental I Incompleto',
        'Ensino Fundamental I Completo',
        'Ensino Fundamental II Incompleto',
        'Ensino Fundamental II Completo',
        'Ensino Médio Incompleto',
        'Ensino Médio Completo',
        'Ensino Superior Incompleto',
        'Ensino Superior Completo',
        'Pós-graduação'
    ]
    
    # Plotar gráfico de barras para média por escolaridade do pai
    plt.figure(figsize=(16, 8))
    media_por_esc_pai = df.groupby('ESCOLARIDADE_PAI')['NU_NOTA_CH'].mean().reset_index()
    media_por_esc_pai['ordem'] = media_por_esc_pai['ESCOLARIDADE_PAI'].map({v: i for i, v in enumerate(ordem_escolaridade)})
    media_por_esc_pai = media_por_esc_pai.sort_values('ordem')
    
    sns.barplot(x='ESCOLARIDADE_PAI', y='NU_NOTA_CH', data=media_por_esc_pai)
    plt.title('Média das Notas de Ciências Humanas por Escolaridade do Pai - ENEM 2023')
    plt.xlabel('Escolaridade do Pai')
    plt.ylabel('Média da Nota')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('media_notas_por_escolaridade_pai.png')
    plt.close()
    
    # Plotar gráfico de barras para média por escolaridade da mãe
    plt.figure(figsize=(16, 8))
    media_por_esc_mae = df.groupby('ESCOLARIDADE_MAE')['NU_NOTA_CH'].mean().reset_index()
    media_por_esc_mae['ordem'] = media_por_esc_mae['ESCOLARIDADE_MAE'].map({v: i for i, v in enumerate(ordem_escolaridade)})
    media_por_esc_mae = media_por_esc_mae.sort_values('ordem')
    
    sns.barplot(x='ESCOLARIDADE_MAE', y='NU_NOTA_CH', data=media_por_esc_mae)
    plt.title('Média das Notas de Ciências Humanas por Escolaridade da Mãe - ENEM 2023')
    plt.xlabel('Escolaridade da Mãe')
    plt.ylabel('Média da Nota')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('media_notas_por_escolaridade_mae.png')
    plt.close()
    
    # Análise por tipo de escola
    print("\nMédia de notas por escolaridade do pai e tipo de escola:")
    media_por_esc_pai_escola = df.groupby(['ESCOLARIDADE_PAI', 'TIPO_ESCOLA_DESC'])['NU_NOTA_CH'].mean().unstack().reset_index()
    print(media_por_esc_pai_escola)
    
    print("\nMédia de notas por escolaridade da mãe e tipo de escola:")
    media_por_esc_mae_escola = df.groupby(['ESCOLARIDADE_MAE', 'TIPO_ESCOLA_DESC'])['NU_NOTA_CH'].mean().unstack().reset_index()
    print(media_por_esc_mae_escola)
    
    # Plotar heatmap de escolaridade dos pais por tipo de escola
    plt.figure(figsize=(16, 10))
    # Filtrar para incluir apenas escolas públicas e privadas
    media_cruzada_pai = df[df['TIPO_ESCOLA_DESC'].isin(['Pública', 'Privada'])].pivot_table(
        index='ESCOLARIDADE_PAI', 
        columns='TIPO_ESCOLA_DESC', 
        values='NU_NOTA_CH', 
        aggfunc='mean'
    )
    # Reordenar o índice
    ordem_presente_pai = [e for e in ordem_escolaridade if e in media_cruzada_pai.index]
    media_cruzada_pai = media_cruzada_pai.reindex(ordem_presente_pai)
    
    sns.heatmap(media_cruzada_pai, annot=True, cmap='YlGnBu', fmt='.1f')
    plt.title('Média das Notas de CH por Escolaridade do Pai e Tipo de Escola')
    plt.tight_layout()
    plt.savefig('heatmap_escolaridade_pai_tipo_escola.png')
    plt.close()
    
    # Plotar heatmap de escolaridade da mãe por tipo de escola
    plt.figure(figsize=(16, 10))
    media_cruzada_mae = df[df['TIPO_ESCOLA_DESC'].isin(['Pública', 'Privada'])].pivot_table(
        index='ESCOLARIDADE_MAE', 
        columns='TIPO_ESCOLA_DESC', 
        values='NU_NOTA_CH', 
        aggfunc='mean'
    )
    # Reordenar o índice
    ordem_presente_mae = [e for e in ordem_escolaridade if e in media_cruzada_mae.index]
    media_cruzada_mae = media_cruzada_mae.reindex(ordem_presente_mae)
    
    sns.heatmap(media_cruzada_mae, annot=True, cmap='YlGnBu', fmt='.1f')
    plt.title('Média das Notas de CH por Escolaridade da Mãe e Tipo de Escola')
    plt.tight_layout()
    plt.savefig('heatmap_escolaridade_mae_tipo_escola.png')
    plt.close()

    # Função para analisar a relação entre acesso à internet e notas
def analisar_internet_notas(df):
    """
    Analisa a relação entre acesso à internet e notas de Ciências Humanas.
    
    Args:
        df: DataFrame com os dados preparados
    """
    print("\n--- ANÁLISE DA RELAÇÃO ENTRE ACESSO À INTERNET E NOTAS ---")
    
    # Estatísticas por acesso à internet
    stats_por_internet = df.groupby('ACESSO_INTERNET')['NU_NOTA_CH'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    
    print("\nEstatísticas de notas por acesso à internet:")
    print(stats_por_internet)
    
    # Plotar boxplot de notas por acesso à internet
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='ACESSO_INTERNET', y='NU_NOTA_CH', data=df)
    plt.title('Notas de Ciências Humanas por Acesso à Internet - ENEM 2023')
    plt.xlabel('Acesso à Internet')
    plt.ylabel('Nota')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('notas_por_acesso_internet.png')
    plt.close()
    
    # Gráfico de barras para média por acesso à internet
    plt.figure(figsize=(10, 6))
    media_por_internet = df.groupby('ACESSO_INTERNET')['NU_NOTA_CH'].mean().reset_index()
    
    sns.barplot(x='ACESSO_INTERNET', y='NU_NOTA_CH', data=media_por_internet)
    plt.title('Média das Notas de Ciências Humanas por Acesso à Internet - ENEM 2023')
    plt.xlabel('Acesso à Internet')
    plt.ylabel('Média da Nota')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('media_notas_por_acesso_internet.png')
    plt.close()
    
    # Teste estatístico para verificar se há diferença significativa
    grupo_com_internet = df[df['ACESSO_INTERNET'] == 'Sim']['NU_NOTA_CH']
    grupo_sem_internet = df[df['ACESSO_INTERNET'] == 'Não']['NU_NOTA_CH']
    
    # Verificar se os grupos têm dados suficientes
    if len(grupo_com_internet) > 0 and len(grupo_sem_internet) > 0:
        t_stat, p_value = stats.ttest_ind(grupo_com_internet, grupo_sem_internet, equal_var=False)
        print(f"\nResultado do teste t para diferença de médias:")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("Há diferença estatisticamente significativa entre os grupos (p < 0.05).")
        else:
            print("Não há diferença estatisticamente significativa entre os grupos (p >= 0.05).")
    else:
        print("\nNão há dados suficientes em um ou ambos os grupos para realizar o teste estatístico.")

# Função para analisar a relação entre tipo de escola e notas
def analisar_tipo_escola_notas(df):
    """
    Analisa a relação entre tipo de escola (pública x privada) e notas de Ciências Humanas.
    
    Args:
        df: DataFrame com os dados preparados
    """
    print("\n--- ANÁLISE DA RELAÇÃO ENTRE TIPO DE ESCOLA E NOTAS ---")
    
    # Estatísticas por tipo de escola
    stats_por_escola = df.groupby('TIPO_ESCOLA_DESC')['NU_NOTA_CH'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    
    print("\nEstatísticas de notas por tipo de escola:")
    print(stats_por_escola)
    
    # Plotar boxplot de notas por tipo de escola
    plt.figure(figsize=(12, 6))
    # Filtrar apenas escolas públicas e privadas para melhor visualização
    df_filtrado = df[df['TIPO_ESCOLA_DESC'].isin(['Pública', 'Privada'])]
    
    sns.boxplot(x='TIPO_ESCOLA_DESC', y='NU_NOTA_CH', data=df_filtrado)
    plt.title('Notas de Ciências Humanas por Tipo de Escola - ENEM 2023')
    plt.xlabel('Tipo de Escola')
    plt.ylabel('Nota')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('notas_por_tipo_escola.png')
    plt.close()
    
    # Analisar por região/UF
    if 'CO_UF_ESC' in df.columns:
        # Mapear códigos de UF para nomes
        mapa_uf = {
            11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO',
            21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN', 25: 'PB', 26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA',
            31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP',
            41: 'PR', 42: 'SC', 43: 'RS',
            50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF'
        }
        df['UF_ESCOLA'] = df['CO_UF_ESC'].map(mapa_uf)
        
        # Agrupar por UF e tipo de escola
        media_por_uf_escola = df.pivot_table(
            index='UF_ESCOLA', 
            columns='TIPO_ESCOLA_DESC',
            values='NU_NOTA_CH',
            aggfunc='mean'
        )
        
        print("\nMédia de notas por UF e tipo de escola:")
        print(media_por_uf_escola)
        
        # Plotar heatmap das médias por UF e tipo de escola
        plt.figure(figsize=(12, 10))
        sns.heatmap(media_por_uf_escola, annot=True, cmap='YlGnBu', fmt='.1f')
        plt.title('Média das Notas de CH por UF e Tipo de Escola')
        plt.tight_layout()
        plt.savefig('heatmap_uf_tipo_escola.png')
        plt.close()
        
        # Diferença entre públicas e privadas por UF
        if 'Pública' in media_por_uf_escola.columns and 'Privada' in media_por_uf_escola.columns:
            media_por_uf_escola['Diferença'] = media_por_uf_escola['Privada'] - media_por_uf_escola['Pública']
            
            plt.figure(figsize=(14, 8))
            media_por_uf_escola['Diferença'].sort_values(ascending=False).plot(kind='bar')
            plt.title('Diferença de Desempenho entre Escolas Privadas e Públicas por UF')
            plt.xlabel('UF')
            plt.ylabel('Diferença na Média de Notas (Privada - Pública)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig('diferenca_privada_publica_por_uf.png')
            plt.close()
    
    # Analisar desempenho por capital x interior
    # Necessário verificar se temos a informação de capital x interior nos dados
    if 'CO_MUNICIPIO_ESC' in df.columns:
        # Lista de códigos de municípios das capitais (você precisará consultar o IBGE para obter esta lista)
        # Exemplo simplificado com algumas capitais
        codigos_capitais = [
            2704302,  # Maceió
            1302603,  # Manaus
            1600303,  # Macapá
            2927408,  # Salvador
            2304400,  # Fortaleza
            5300108,  # Brasília
            3205309,  # Vitória
            5208707,  # Goiânia
            2111300,  # São Luís
            5103403,  # Cuiabá
            1501402,  # Belém
            2507507,  # João Pessoa
            4106902,  # Curitiba
            2611606,  # Recife
            2204202,  # Teresina
            3304557,  # Rio de Janeiro
            2408102,  # Natal
            4314902,  # Porto Alegre
            1100205,  # Porto Velho
            1400100,  # Boa Vista
            4205407,  # Florianópolis
            3550308,  # São Paulo
            2800308,  # Aracaju
            1721000   # Palmas
        ]
        
        df['ESCOLA_CAPITAL'] = df['CO_MUNICIPIO_ESC'].isin(codigos_capitais)
        
        # Analisar desempenho por capital x interior e tipo de escola
        media_capital_interior = df.pivot_table(
            index='ESCOLA_CAPITAL', 
            columns='TIPO_ESCOLA_DESC',
            values='NU_NOTA_CH',
            aggfunc='mean'
        )
        
        print("\nMédia de notas por Capital x Interior e tipo de escola:")
        print(media_capital_interior)
        
        # Plotar gráfico de barras comparando capital x interior por tipo de escola
        plt.figure(figsize=(12, 6))
        media_capital_interior.plot(kind='bar')
        plt.title('Média das Notas de CH por Capital x Interior e Tipo de Escola')
        plt.xlabel('Escola em Capital')
        plt.xticks([0, 1], ['Interior', 'Capital'], rotation=0)
        plt.ylabel('Média da Nota')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Tipo de Escola')
        plt.tight_layout()
        plt.savefig('media_notas_capital_interior_tipo_escola.png')
        plt.close()

# Função para analisar a relação entre localização da escola (urbana x rural) e notas
def analisar_localizacao_escola_notas(df):
    """
    Analisa a relação entre localização da escola (urbana x rural) e notas de Ciências Humanas.
    
    Args:
        df: DataFrame com os dados preparados
    """
    print("\n--- ANÁLISE DA RELAÇÃO ENTRE LOCALIZAÇÃO DA ESCOLA E NOTAS ---")
    
    # Estatísticas por localização da escola
    stats_por_localizacao = df.groupby('LOCALIZACAO_ESCOLA_DESC')['NU_NOTA_CH'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    
    print("\nEstatísticas de notas por localização da escola:")
    print(stats_por_localizacao)
    
    # Plotar boxplot de notas por localização da escola
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='LOCALIZACAO_ESCOLA_DESC', y='NU_NOTA_CH', data=df)
    plt.title('Notas de Ciências Humanas por Localização da Escola - ENEM 2023')
    plt.xlabel('Localização da Escola')
    plt.ylabel('Nota')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('notas_por_localizacao_escola.png')
    plt.close()
    
    # Gráfico de barras para média por localização da escola
    plt.figure(figsize=(10, 6))
    media_por_localizacao = df.groupby('LOCALIZACAO_ESCOLA_DESC')['NU_NOTA_CH'].mean().reset_index()
    
    sns.barplot(x='LOCALIZACAO_ESCOLA_DESC', y='NU_NOTA_CH', data=media_por_localizacao)
    plt.title('Média das Notas de Ciências Humanas por Localização da Escola - ENEM 2023')
    plt.xlabel('Localização da Escola')
    plt.ylabel('Média da Nota')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('media_notas_por_localizacao_escola.png')
    plt.close()
    
    # Analisar por tipo de escola e localização
    media_por_tipo_localizacao = df.pivot_table(
        index='TIPO_ESCOLA_DESC', 
        columns='LOCALIZACAO_ESCOLA_DESC',
        values='NU_NOTA_CH',
        aggfunc='mean'
    )
    
    print("\nMédia de notas por tipo de escola e localização:")
    print(media_por_tipo_localizacao)
    
    # Plotar gráfico de barras agrupado
    plt.figure(figsize=(12, 6))
    media_por_tipo_localizacao.plot(kind='bar')
    plt.title('Média das Notas de CH por Tipo de Escola e Localização')
    plt.xlabel('Tipo de Escola')
    plt.ylabel('Média da Nota')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Localização da Escola')
    plt.tight_layout()
    plt.savefig('media_notas_tipo_escola_localizacao.png')
    plt.close()

# Função para analisar a relação entre UF/estado da escola e notas
def analisar_uf_escola_notas(df):
    """
    Analisa a relação entre UF/estado da escola e notas de Ciências Humanas.
    
    Args:
        df: DataFrame com os dados preparados
    """
    print("\n--- ANÁLISE DA RELAÇÃO ENTRE UF/ESTADO DA ESCOLA E NOTAS ---")
    
    if 'CO_UF_ESC' in df.columns:
        # Mapear códigos de UF para nomes
        mapa_uf = {
            11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO',
            21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN', 25: 'PB', 26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA',
            31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP',
            41: 'PR', 42: 'SC', 43: 'RS',
            50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF'
        }
        df['UF_ESCOLA'] = df['CO_UF_ESC'].map(mapa_uf)
        
        # Estatísticas por UF
        stats_por_uf = df.groupby('UF_ESCOLA')['NU_NOTA_CH'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
        stats_por_uf = stats_por_uf.sort_values(by='mean', ascending=False)
        
        print("\nEstatísticas de notas por UF da escola:")
        print(stats_por_uf)
        
        # Plotar gráfico de barras para média por UF
        plt.figure(figsize=(16, 8))
        sns.barplot(x='UF_ESCOLA', y='mean', data=stats_por_uf)
        plt.title('Média das Notas de Ciências Humanas por UF da Escola - ENEM 2023')
        plt.xlabel('UF da Escola')
        plt.ylabel('Média da Nota')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('media_notas_por_uf_escola.png')
        plt.close()
        
        # Boxplot por UF
        plt.figure(figsize=(18, 10))
        sns.boxplot(x='UF_ESCOLA', y='NU_NOTA_CH', data=df)
        plt.title('Notas de Ciências Humanas por UF da Escola - ENEM 2023')
        plt.xlabel('UF da Escola')
        plt.ylabel('Nota')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('boxplot_notas_por_uf_escola.png')
        plt.close()
        
        # Análise de variância (ANOVA) para verificar se há diferença entre os estados
        if len(df['UF_ESCOLA'].dropna().unique()) > 1:  # Verificar se há mais de um estado
            from scipy.stats import f_oneway
            
            # Criar listas para armazenar as amostras de cada UF
            amostras_uf = []
            ufs = df['UF_ESCOLA'].dropna().unique()
            
            for uf in ufs:
                amostra = df[df['UF_ESCOLA'] == uf]['NU_NOTA_CH'].dropna()
                if len(amostra) > 30:  # Considerar apenas UFs com amostra significativa
                    amostras_uf.append(amostra)
            
            if len(amostras_uf) > 1:  # Verificar se temos pelo menos 2 amostras
                f_stat, p_value = f_oneway(*amostras_uf)
                print("\nResultado da ANOVA para diferença entre UFs:")
                print(f"F-statistic: {f_stat:.4f}")
                print(f"p-value: {p_value:.4f}")
                
                if p_value < 0.05:
                    print("Há diferença estatisticamente significativa entre as UFs (p < 0.05).")
                else:
                    print("Não há diferença estatisticamente significativa entre as UFs (p >= 0.05).")
            else:
                print("\nNão há amostras suficientes para realizar o teste ANOVA.")
    else:
        print("\nA coluna 'CO_UF_ESC' não está presente nos dados.")

# Função para analisar a relação entre localização da prova e notas
def analisar_municipio_prova_notas(df):
    """
    Analisa a relação entre município de aplicação da prova e notas de Ciências Humanas.
    
    Args:
        df: DataFrame com os dados preparados
    """
    print("\n--- ANÁLISE DA RELAÇÃO ENTRE MUNICÍPIO DE APLICAÇÃO DA PROVA E NOTAS ---")
    
    if 'CO_MUNICIPIO_PROVA' in df.columns:
        # Agrupar por município de prova
        media_por_municipio = df.groupby('CO_MUNICIPIO_PROVA')['NU_NOTA_CH'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
        
        # Ordenar por média descendente
        top_municipios = media_por_municipio.sort_values(by='mean', ascending=False).head(20)
        bottom_municipios = media_por_municipio.sort_values(by='mean').head(20)
        
        print("\nTop 20 municípios com maior média de notas:")
        print(top_municipios)
        
        print("\nTop 20 municípios com menor média de notas:")
        print(bottom_municipios)
        
        # Municípios com maior variação
        media_por_municipio['cv'] = media_por_municipio['std'] / media_por_municipio['mean']  # Coeficiente de variação
        
        top_var_municipios = media_por_municipio.sort_values(by='cv', ascending=False).head(20)
        bottom_var_municipios = media_por_municipio.sort_values(by='cv').head(20)
        
        print("\nTop 20 municípios com maior variabilidade de notas (maior CV):")
        print(top_var_municipios)
        
        print("\nTop 20 municípios com menor variabilidade de notas (menor CV):")
        print(bottom_var_municipios)
        
        # Plotar histograma das médias por município
        plt.figure(figsize=(12, 6))
        sns.histplot(media_por_municipio['mean'], kde=True, bins=30)
        plt.title('Distribuição das Médias de Notas CH por Município de Prova - ENEM 2023')
        plt.xlabel('Média da Nota')
        plt.ylabel('Frequência')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('distribuicao_medias_por_municipio_prova.png')
        plt.close()
        
        # Se tivermos dados suficientes, podemos tentar correlacionar com IDH ou outros indicadores
        # (Isso exigiria dados externos, que não estão disponíveis no script atual)
        print("\nNota: Para correlacionar as notas com IDH ou outros indicadores socioeconômicos, ")
        print("seria necessário incorporar dados externos sobre os municípios.")
    else:
        print("\nA coluna 'CO_MUNICIPIO_PROVA' não está presente nos dados.")

# Função principal para executar todas as análises
def main():
    """
    Função principal que coordena a execução de todas as análises.
    """
    try:
        # Verificar se os dados já foram carregados
        if 'dados' in globals():
            print("Utilizando dados já carregados")
            df = dados
        else:
            print("Dados não encontrados. Por favor, carregue os dados primeiro.")
            return
        
        # Preparar os dados
        print("\nPreparando os dados para análise...")
        df_clean = preparar_dados(df)
        print(f"Dados preparados. Shape após limpeza: {df_clean.shape}")
        
        # Executar análises
        analisar_distribuicao_notas(df_clean)
        analisar_idade_notas(df_clean)
        analisar_renda_notas(df_clean)
        analisar_escolaridade_pais_notas(df_clean)
        analisar_internet_notas(df_clean)
        analisar_tipo_escola_notas(df_clean)
        analisar_localizacao_escola_notas(df_clean)
        analisar_uf_escola_notas(df_clean)
        analisar_municipio_prova_notas(df_clean)
        
        print("\nTodasas análises foram concluídas com sucesso!")
        
    except Exception as e:
        print(f"Erro durante a execução das análises: {e}")
        import traceback
        traceback.print_exc()

# Executar o script se for o script principal
if __name__ == "__main__":
    main()