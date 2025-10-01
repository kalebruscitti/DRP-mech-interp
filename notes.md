# Notes on **Progress Measures for Grokking via Mechanistic Interpretability**

## Flow of the Model

1. **Input as one-hot vectors**  
   - Each token in the input $(a, b, =)$ is turned into a one-hot vector of length 113.  
   - These form the matrix:  
     $X ∈ ℝ^{3 × 113}$

2. **Embedding**  
   - Multiply by the embedding matrix:  
     $W_E ∈ ℝ^{113 × 128}$
   - Result:  
     $X · W_E ∈ ℝ^{3 × 128}$ 
     → three 128-dimensional embedded vectors $(v_a, v_b, v_=)$.

3. **Transformer Block**  
   - **Attention phase:**  
     - Each embedding is projected into queries, keys, and values:  
       $Q = XW_Q,   K = XW_K,   V = XW_V$
       with $W_Q, W_K, W_V ∈ ℝ^{128 × 32}$ per head (in our case, we have 4 attention heads).  
     - Attention scores:  
       $softmax((QK^T) / sqrt(d_{head}))V $
       → the "=" token attends to both a and b and collects their information.

   - **MLP phase:**  
     - Each updated token vector goes through a two-layer feedforward network:  
       $MLP(x) = \sigma(xW_{in}) W_{out}$
       where $W_{in} ∈ ℝ^{(128 × d_{mlp})}, W_{out} ∈ ℝ^{(d_{mlp} × 128)}$.  
     - This step encodes the addition operation in the hidden representation.

4. **Final hidden vector**  
   - At the "=" position, the hidden vector now encodes the result of $(a + b \mod 113)$.

5. **Unembedding**  
   - Multiply by the unembedding matrix:  
     W_U ∈ ℝ^{128 × 113}
   - Produces logits:  
     logits ∈ ℝ^{113}

6. **Softmax & Prediction**  
   - Apply softmax to convert logits into probabilities.  
   - The highest-probability token is selected as the model’s prediction for $(a + b \mod 113)$.

