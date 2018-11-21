# byte_cup
The branch contains code for the bytecup.
For more information about bytecup, please check https://biendata.com/competition/bytecup2018/.
The bytecup is a competition about automatic summarisation of English articles.

My approach is purely abstractive.

Features include:

1. bi-directional gru for encoder
2. attention for encoder output
3. beam-search decode
4. pointer-generator
5. coverage
6. embedding sharing between encoder input, decoder input and softmax output.
