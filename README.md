## abordagem da tokenização

##### formato:

```
[CLS] question [SEP] code [SEP]
```

- **[CLS]** (Classification Token): Token inicial que serve como agregador de informações para toda a sequência. Durante o fine-tuning, a representação vetorial deste token é utilizada como entrada para a camada de classificação que determina a presença de vulnerabilidades.
- **question**: sequência textual em linguagem natural que contextualiza a análise de segurança.
- **[SEP]** (Separator Token): Delimitador que estabelece a fronteira entre diferentes componentes da sequência, permitindo ao modelo diferenciar entre a pergunta e o código-fonte.
- **code**: sequência de tokens representando o código-fonte a ser analisado quanto à presença de vulnerabilidades.

Esta estrutura de sequência é fundamental para o processo de atenção cruzada (cross-attention) entre a pergunta e o código durante o processamento pelo modelo, permitindo que o contexto da pergunta influencie a interpretação do código.

### processo de tokenização

O processo de tokenização é implementado utilizando os seguintes parâmetros técnicos:

```python
encoded = tokenizer(
    question,
    code_text,
    padding='max_length',
    truncation='only_second',
    max_length=512,
    return_tensors='pt',
    return_overflowing_tokens=True,
    stride=128
)
```

Onde:

- **padding='max_length'**: Padroniza todas as sequências para um comprimento fixo de 512 tokens, adicionando tokens de padding quando necessário.
- **truncation='only_second'**: Implementa uma estratégia de truncamento que preserva integralmente a pergunta e trunca apenas o código-fonte quando a sequência excede o limite máximo.
- **max_length=512**: Define o comprimento máximo da sequência em 512 tokens, conforme as especificações do modelo CodeBERT.
- **return_tensors='pt'**: Especifica o formato de saída como tensores PyTorch, compatíveis com a infraestrutura de treinamento.
- **return_overflowing_tokens=True**: Habilita o processamento de sequências que excedem o limite máximo através da técnica de janela deslizante.
- **stride=128**: Define o tamanho da sobreposição entre janelas consecutivas como 128 tokens, garantindo continuidade contextual.

### tratamento de sequências extensas

Para código-fonte que excede o limite máximo de 512 tokens, implementamos um mecanismo de processamento baseado em sliding window com sobreposição. Este mecanismo opera da seguinte forma:

1. A sequência original é segmentada em múltiplas subsequências com sobreposição de 128 tokens entre segmentos adjacentes.
2. Cada subsequência é processada independentemente pelo modelo.
3. As representações resultantes são posteriormente agregadas através de um mecanismo de pooling para produzir uma classificação final.

Este método permite a análise de arquivos de código-fonte arbitrariamente extensos, mantendo a integridade contextual através das sobreposições entre segmentos adjacentes.



## representação da pipeline (?)

```
+-----------------------------+    +---------------------------+
| Entrada:                    | -> | Saida:                    |
| - código                    |    | - classificação binária   |                
+-----------------------------+    +---------------------------+
            |                                    ^
            v                                    |
+-----------------------------+    +---------------------------+
| Tokenização                 | -> | CodeBERT                  |          
+-----------------------------+    +---------------------------+
```



## cool papers

1. Feng, Z., Guo, D., Tang, D., Duan, N., Feng, X., Gong, M., Shou, L., Qin, B., Liu, T., Jiang, D., & Zhou, M. (2020). CodeBERT: A Pre-Trained Model for Programming and Natural Languages. Findings of EMNLP.

2. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv preprint arXiv:1907.11692.

3. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems.

---

