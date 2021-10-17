# dekorde
트랜스포머 밑바닥부터 구현 -  @Sesame's Pirates

### Notations
Note that all vector outputs inside the network have same dimension, E, with embedding to make skip connection & self-attention work

- E: embedding/hidden dimension
- B: batch size
- L: maximum length of the input
- 


## Implemented
### Scheduler & Dropout
- Scheduler: Scheduler based on step numbers
- Dropout: In training phase, authors mentioned that dropout with probability=0.1 applied to all layers in the tranformer block & to the embedding vector with positional encodings

### Multi head attention


### Encoder
Encoder block
- Self-attention layer
- Skip-connection
- Feed-forward network

### Decoder


## Training Phase
### Dataset
- Training data with similar lengths are batched to be transformers' input
- Dataset was tokenized by pretrained BERT, and this set of tokens form vocabulary for our model
- Vocabulary is for both Gibberish & Real sentences



## Test phase