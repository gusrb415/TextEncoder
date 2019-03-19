# TextEncoder
Text encoder for predicting the next word using Keras framework

1. Run python main.py -mode train -epochs 1 -batch_size 32 -embedding_dim 100 -hidden_size 500 -drop 0.5 -student_id 12345678 -saved_model models/model.h5
2. Run python main.py -mode test -saved_model models/model.h5 -input data/valid.csv -student_id 12345678
3. Run python scorer.py -submission 12345678_valid_result.csv
