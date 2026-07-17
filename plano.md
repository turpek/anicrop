## Lista de Tarefas

1. Usar o `Zarr` para dar suporte a imagens grandes.
2. Implementar uma classe que renderiza o `Layer` no espaço local do layer
3. Criar um novo sistema de **LOD**, mas agora na classe `EditLayer`.
4. Migrar para o sistema de **tiles** com a classe `Tile` que vai compor a classe `Layer`.
5. Criar uma classe para renderizar o `Layer` no espaço da viewport (olhar com carinho para final_region e render_region).
6. Alocar um único buffer para renderizar os edits no processo de população das tiles


---

### Zarr no sistema

A classe `Image` espera receber um `ndarray`, então o sistema já suporta o `Zarr` nativamente, o que temos que fazer é definir quem cria esse objeto e quando criar, propostas:

- Opção A: `open`função/método que abre arquivos
- Opção B: `EditLayer` classe que recebe o `Image`

A opção A parece tentadora e é fácil de implementar, mas não é generalista, pois se criarmos um `ndarray` em memória teriamos que criar uma nova função para fazer a conversão, já a opção B da mais responsabilidade para o `EditLayer` lidar, mas centraliza tudo nele o que é uma boa coisa, já que ele se torna a unica fonte de criação dos `Zarr`, no caso a conversão se torna opcional (a ver se colocamos regras para impedir a criação de *zarrs* de imagens de tamanho comum).

### Renderização no espaço local do layer

Quando criei (2), eu citei o espaço local do layer, mas eu tomei a liberdade de alterar para o espaço do layer, pois o `Layer` foi desenhado para se moldar ao primeiro `EditLayer` inserido, então essa sutil mudança, nós permite em um futuro separar a dependência do layer com o edit, de forma a não quebrar nenhum contrato.

Sobre a implementação, na criação dos **edits** injetamos a matriz inversa que os leva para o espaço do **layer**, então passamos a responsabilidade de

### Implementação do `LOD`

A implementação do novo sistema de `LOD` deve seguir as seguintes regras

1. Para imagens convertidas em "Zarr" usar cache.
2. Para imagens produzidas pelo "ndarray" usar o resize com a interpolação INTER_AREA.
3. O nível de LOD é definido como N = 2^(-n) onde n = 0, 1, 2, ....
4. A variável n é calculada a partir do fator de escala f como n = floor(-log2(f)).
5. Se o fator de escala for maior que 1 retornamos a região de interesse da imagem original

O acesso do `LOD` se da por um método na classe `EditLayer`, usando as regras definidas acima
