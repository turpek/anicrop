# anicrop

**anicrop** é um motor (core) de composição e edição de imagens 2D de alta performance para Python. Projetado para ser não-destrutivo e matematicamente preciso, o motor utiliza uma arquitetura baseada em camadas, transformações imutáveis e um pipeline de renderização otimizado por caches de múltiplos estágios.



## 🚀 Principais Funcionalidades

- **Composição Não-Destrutiva**: As imagens originais nunca são alteradas; todas as edições e filtros são aplicados dinamicamente em uma pilha de renderização.

- **Normalização de Edits via Matriz Inversa**: Ao adicionar um `EditLayer`, o sistema utiliza a matriz inversa da transformação atual do Layer para "normalizar" o asset. Isso garante que, ao aplicar a transformação final, o edit seja levado precisamente para a posição visual desejada.

- **Cache de Dois Estágios**: Separa a composição local (edits e filtros) da transformação espacial global (giros e escalas), permitindo navegação fluida em tempo real.

- **Blend Modes Robustos**: Suporte a diversos modos de mesclagem com tratamento rigoroso de canais Alpha para evitar artefatos visuais.
  
  **Sistemas Híbridos de Transformação**:
  
  - **`Transform`**: Interface imutável e preguiçosa (*Lazy Evaluation*). Armazena uma lista de operações (Rotação, Escala, Translação) que só são "esmagadas" em uma matriz final no momento em que o cliente seta o objeto no Layer.
  
  - **`TransformComposer`**: Motor interno de transformações com API fluida. Aplica a multiplicação de matrizes de forma imediata (*Eager*) para manipulação direta dentro do Layer

## 🏗️ Arquitetura do Sistema

O pipeline de renderização segue uma ordem estrita para garantir a consistência entre transformações espaciais e efeitos de valor:

1. **Composição Espacial**: Aplicação das matrizes de transformação em cada `EditLayer` e na imagem base, unificando os resultados para formar a imagem final da camada.

2. **Pós-Processamento de Valor**: Aplicação de filtros e ajustes de cor diretamente sobre a imagem final composta da camada.

3. **Gerenciamento de Cache**: O `LayerRender` monitora flags de "sujeira" para decidir se reaproveita a composição local ou se precisa reconstruir o quadro a partir dos assets.

## 🛠️ Tecnologias Utilizadas

- **Linguagem**: Python

- **Processamento Numérico**: NumPy (representação interna via `ndarray`)

- **Visão Computacional**: OpenCV (Warping, Filtragem, Espaços de Cor)

- **Workflow**: Recomendado o uso de `uv` para gestão de dependências.

## 📋 Roadmap de Desenvolvimento

- [ ] Implementação da classe `ImmutableTransform` para avaliação preguiçosa.

- [ ] Finalização do pipeline de cache no `LayerRender`.

- [ ] Introdução de `AdjustmentLayers` para filtros globais com máscaras.

- [ ] Integração completa da lógica de `merge_down` na `LayerStack`.
