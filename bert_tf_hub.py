import bert
import numpy as np
import tensorflow as tf
from tensorflow_hub import KerasLayer

def create_tokenizer(bert_model, trainable=False):
    """ Create Bert tokenizer instance. """

    bert_layer = KerasLayer(bert_model, trainable=trainable)
    BertTokenizer = bert.bert_tokenization.FullTokenizer
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocab_file, do_lower_case)
    return tokenizer

def get_masks(tokens, max_seq_length):
    """ Create mask for padding. """
    
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def get_segments(tokens, max_seq_length):
    """ Create segmentation: 
        0 for the first sequence,
        1 for the second, etc. """

    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
    """ Create token ids from tokenizer vocab. """
    
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def build_inputs(sequences, tokenizer, max_seq_length):
    """ Convert sequences into Bert inputs (word ID, mask, type ID). """
    
    input_word_ids, input_mask, input_type_ids = [], [], []
    for sequence in sequences:
        sequence_tokens = tokenizer.tokenize(sequence)
        sequence_tokens = ["[CLS]"] + sequence_tokens + ["[SEP]"]

        sequence_ids = get_ids(sequence_tokens, tokenizer, max_seq_length)
        input_word_ids.append(sequence_ids)

        sequence_mask = get_masks(sequence_tokens, max_seq_length)
        input_mask.append(sequence_mask)

        sequence_segments = get_segments(sequence_tokens, max_seq_length)
        input_type_ids.append(sequence_segments)
    
    transformed_seq = dict(
        input_word_ids=tf.convert_to_tensor(np.asarray(input_word_ids).astype('int32'), dtype=tf.int32),
        input_mask=tf.convert_to_tensor(np.asarray(input_mask).astype('int32'), dtype=tf.int32),
        input_type_ids=tf.convert_to_tensor(np.asarray(input_type_ids).astype('int32'), dtype=tf.int32)
    )
    
    return transformed_seq

def get_model_inputs():
    """ Initialize Bert inputs. """
    
    inputs = dict(
      input_word_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
      input_mask=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
      input_type_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
    )
    return inputs

def get_embedding(bert_model, trainable=True):
    """ Create Bert embedding layer. """

    bert_layer = KerasLayer(bert_model, trainable=trainable, name='bert_embedding')
    return bert_layer


# references :
# https://towardsdatascience.com/simple-bert-using-tensorflow-2-0-132cb19e9b22