# Recurrent Neural Network

## LSTM: Long-short term memory cells
One of the most fundamental works, LSTM is the dominant architecture in RNNs.
1. the input gate: $i_t$
2. the forget gate: $f_t$
3. the new cell/context vector: $c_t$
4. the almost new output: $o_t$
5. the new context: $h_t$

## Keras LSTM layer
- fixed length of sequence,  
- return_sequences by default is False, it will flatten the output


## Notes
1. RNN-based architectures used to work very well especially with LSTM and GRU components. The problem is only for small sequences (< 20 time steps). The limitations of RNN: 1) the intermediate representation z cannot encode information from all the input timesteps, this is commonly known as the bottleneck problem; 2) The stacked RNN layer usually create the well-know vanishing gradient problem. Attention was born to address these problems on the Seq2Seq model.

## References
[1] https://theaisummer.com/understanding-lstm/
[2] https://npitsillos.github.io/blog/2021/recurrent-ppo/
