# POSTECH MLE - Módulo 4: Processamento de Linguagem Natural

Coleção de notebooks, imagens e roteiros utilizados nas aulas de NLP do curso de Machine Learning Engineering (MLE) da POSTECH. O material conecta fundamentos teóricos e implementações práticas com `scikit-learn`, `NLTK`, `gensim`, `PyTorch`, `Transformers` e ferramentas de avaliação.

## Estrutura
- `LICENSE.txt`: termos da licença MIT utilizados neste módulo.
- `img.zip`: pacote de figuras usadas nos notebooks (`pipeline`, `bert_tasks`, etc.). Extraia com `unzip img.zip` para criar a pasta `img/`.
- `AprendizadoSupervisionado.zip`: conjunto adicional de datasets/imagens de apoio. Descompacte apenas se quiser reutilizar os arquivos de referência contidos nele.
- `MLE-Aula1-Pre-processamentoDeTextos.ipynb`: pipeline completo de limpeza, tokenização, regex, remoção de *stopwords*, stemming/lemmatização e avaliação de *POS-taggers* em português via NLTK.
- `MLE-Aula2-Embeddings.ipynb`: fundamentos de embeddings, comparação CBOW vs. Skip-gram e treino de um `Word2Vec` com `gensim` sobre o dataset UCI News Aggregator.
- `MLE-Aula2_e_4-AnaliseDeSentimento.ipynb`: aula dupla que mostra engenharia de atributos com Bag-of-Words e TF-IDF, construção de um classificador linear (SVM) para notícias e propõe um exercício guiado de análise de sentimentos com tweets.
- `MLE-Aula3-BERT.ipynb`: revisão conceitual do BERT, tokenização, arquitetura e *fine-tuning* de um `DistilBERT` para classificar o dataset BBC News usando PyTorch + Transformers.
- `MLE-Aula5-NER.ipynb`: introdução a NER, uso de modelos pré-treinados (NLTK, spaCy, BERTimbau) e *training loop* completo de um `XLM-RoBERTa` sobre recortes do dataset PAN-X/XTREME com avaliação via `seqeval`.
- `POSTECH - Aula X - ....pdf`: apresentações em PDF de cada aula (1 a 5) com os mesmos tópicos dos notebooks.

## Conteúdo das aulas
- **Aula 01 – Pré-processamento de textos:** visão geral do pipeline de NLP, criação de *n*-gramas com `CountVectorizer`, tokenização com `word_tokenize`, limpeza com regex, remoção de *stopwords*, stemming (Porter e RSLP) e lematização. Inclui comparação entre Default/Unigram/Bigram taggers treinados com o corpus *Floresta* para português.
- **Aula 02 – Embeddings:** motivação dos word embeddings, como explorar matrizes densas, recursos pré-treinados (Wikipedia2Vec) e implementação prática de `Word2Vec` com `gensim`, ajustando hiperparâmetros e convertendo sentenças em vetores médios para classificação.
- **Aula 02 & 04 – Classificação de texto e Análise de Sentimentos:** etapas para transformar notícias do `uci-news-aggregator.csv` em atributos (Bag-of-Words/TF-IDF), *train/test split*, treinamento de um SVM linear e avaliação com `sklearn.metrics`. A parte de sentimentos propõe a criação de um classificador para tweets (negativo/neutro/positivo) reaproveitando o pipeline apresentado.
- **Aula 03 – BERT:** detalhamento do *pre-training*/ *fine-tuning*, tokens especiais, *attention masks* e estrutura interna do BERT. O notebook cria um `Dataset` customizado para o BBC News, monta um `DataLoader`, define a cabeça de classificação `DistilBertClassification`, treina, salva e reusa o modelo.
- **Aula 05 – NER:** explica o que são entidades PER/ORG/LOC, usa `pt_core_news_sm` (spaCy) e pipelines pré-treinadas (`monilouise/ner_pt_br`) antes de montar um fluxo completo com Hugging Face `datasets`: seleção de idiomas (pt/en/fr) do PAN-X, tokenização alinhada, *Trainer* + `DataCollatorForTokenClassification`, métricas com `seqeval` e análise de erros em nível de token e sequência.

## Requisitos e preparação
- Python 3.10+ com `pip` atualizado.
- Bibliotecas principais: `jupyter`, `numpy`, `pandas`, `scikit-learn`, `nltk`, `gensim`, `matplotlib`, `torch`, `transformers`, `datasets`, `seqeval`, `spacy`, `tqdm`.
- Opcional (mas recomendado) GPU com suporte a CUDA para acelerar o fine-tuning do DistilBERT e do XLM-RoBERTa.
- Acesso à internet para baixar modelos/corpora do NLTK, spaCy, Hugging Face e datasets externos (BBC News, UCI News Aggregator, PAN-X/XTREME, tweets).

### Setup (venv + dependências)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install jupyter numpy pandas scikit-learn nltk gensim matplotlib \
            torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers[torch] datasets seqeval spacy tqdm
```

### Downloads adicionais
```bash
python -m nltk.downloader punkt stopwords wordnet brown floresta averaged_perceptron_tagger
python -m spacy download pt_core_news_sm
```
- Para o dataset BBC News: baixe `bbc-text.csv` em https://www.kaggle.com/datasets/sainijagjit/bbc-dataset e salve em `bases/bbc-text.csv`.
- Para o UCI News Aggregator: baixe em https://archive.ics.uci.edu/ml/datasets/News+Aggregator (ou equivalente no Kaggle) e salve em `Bases/uci-news-aggregator.csv` (note a pasta `Bases/` usada nos notebooks).
- Para os exercícios de tweets, utilize qualquer dataset rotulado (ex.: [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)) e mantenha o caminho consistente com o notebook.

### Dados e assets
```bash
unzip img.zip          # cria a pasta img/ usada por todos os notebooks
unzip AprendizadoSupervisionado.zip
mkdir -p bases Bases   # alguns notebooks usam nomes diferentes
```
> Caso esteja em ambiente com controle de espaço, descompacte apenas os arquivos necessários.

## Como executar os exemplos
- Inicie o ambiente Jupyter com `jupyter lab` (ou `jupyter notebook`) dentro do diretório do módulo e abra o notebook desejado.
- Execute cada notebook sequencialmente: as células já estão ordenadas de forma didática e trazem textos de apoio, desafios e blocos de código referenciando os datasets acima.
- BERT / NER: ajuste `batch_size`, `num_epochs` e `gradient_accumulation_steps` caso esteja sem GPU ou com pouca memória. Você pode apontar o cache da Hugging Face usando `export HF_HOME=./.hf_cache`.
- NLTK / spaCy: confirme que os downloads foram concluídos antes de executar as células com `nltk.download(...)` ou `spacy.load(...)`.
- Para reutilizar modelos fine-tunados, use as células de salvamento/carregamento presentes nos notebooks (`torch.save`, `model.load_state_dict` ou `Trainer.save_model`).

## Boas práticas
- Reduza o tamanho de amostras ou o número de épocas quando estiver apenas demonstrando código ao vivo.
- Tenha em mente que `datasets.load_dataset('xtreme', ...)` pode baixar vários idiomas; use a lista `langs` do notebook para limitar o escopo.
- Para evitar estouros de VRAM, defina `CUDA_VISIBLE_DEVICES=""` ou rode em CPU. Alternativamente, diminua `max_length` e use `fp16` apenas se tiver GPU compatível.
- Os notebooks assumem caminhos relativos; mantenha a estrutura `Bases/`/`bases/` e `img/` na raiz.
- Use `mlflow` ou outra ferramenta de monitoramento se quiser rastrear experimentos, mas lembre-se de ajustar `tracking_uri` antes de compartilhar resultados.

## Licença
Distribuído sob a licença MIT. Consulte `LICENSE.txt` para detalhes completos.
