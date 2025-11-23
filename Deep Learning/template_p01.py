import numpy as np

def softmax(vector):
    '''
    vector: np.array of shape (n, m)
    
    return: np.array of shape (n, m)
        Matrix where softmax is computed for every row independently
    '''
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_

def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    W_mult: np.array of shape (n_features_dec, n_features_enc)
    
    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    # your code here
    # = (n_features_dec, n_states)
    intermediate = np.dot(W_mult, encoder_hidden_states)
    
    # decoder_hidden_state.T * intermediate: (1, n_features_dec) * (n_features_dec, n_states) = (1, n_states)
    attention_scores = np.dot(decoder_hidden_state.T, intermediate)
    
    # Применяем softmax к attention scores
    weights = softmax(attention_scores)  # (1, n_states)
    
    # Вычисляем взвешенную сумму состояний энкодера
    # encoder_hidden_states: (n_features_enc, n_states)
    # weights: (1, n_states)
    # Результат: (n_features_enc, 1)
    attention_vector = np.dot(encoder_hidden_states, weights.T)
    
    return attention_vector

def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    v_add: np.array of shape (n_features_int, 1)
    W_add_enc: np.array of shape (n_features_int, n_features_enc)
    W_add_dec: np.array of shape (n_features_int, n_features_dec)
    
    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    # your code here
        # Вычисляем W_add_enc * h_i для всех h_i
    # W_add_enc: (n_features_int, n_features_enc) * encoder_hidden_states: (n_features_enc, n_states) = (n_features_int, n_states)
    enc_part = np.dot(W_add_enc, encoder_hidden_states)
    
    # Вычисляем W_add_dec * s
    # W_add_dec: (n_features_int, n_features_dec) * decoder_hidden_state: (n_features_dec, 1) = (n_features_int, 1)
    dec_part = np.dot(W_add_dec, decoder_hidden_state)
    
    # Складываем, расширяя dec_part до размеров enc_part
    # dec_part: (n_features_int, 1) -> (n_features_int, n_states) путем повторения столбца
    combined = enc_part + dec_part
    
    # Применяем tanh
    tanh_combined = np.tanh(combined)
    
    # Умножаем на v_add^T: v_add.T: (1, n_features_int) * tanh_combined: (n_features_int, n_states) = (1, n_states)
    attention_scores = np.dot(v_add.T, tanh_combined)
    
    # Применяем softmax к attention scores
    weights = softmax(attention_scores)  # (1, n_states)
    
    # Вычисляем взвешенную сумму состояний энкодера
    # encoder_hidden_states: (n_features_enc, n_states)
    # weights: (1, n_states)
    # Результат: (n_features_enc, 1)
    attention_vector = np.dot(encoder_hidden_states, weights.T)

    return attention_vector
