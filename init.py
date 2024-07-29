from tensorflow import keras

from proteinbert import load_pretrained_model, finetune, evaluate_by_len, FinetuningModelGenerator
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from util.load_dataset import load_dataset, OUTPUT_SPEC, print_performance
from model import AttentionBasedModelGenerator, CNNBasedModelGenerator

# 1. prepare dataset
train_set = load_dataset("myDatasets/training_set.fasta")
valid_set = load_dataset("myDatasets/validation_set.fasta")
# train_set = load_dataset("myDatasets/ldr_training_set.fasta")
# valid_set = load_dataset("myDatasets/ldr_validation_set.fasta")
# train_set = load_dataset("myDatasets/sdr_training_set.fasta")
# valid_set = load_dataset("myDatasets/sdr_validation_set.fasta")

# 2. Loading the pre-trained model
pretrained_model_generator, input_encoder = load_pretrained_model()

# 3. Train and fine-tuning the entire model
training_callbacks = [
    keras.callbacks.ReduceLROnPlateau(patience=1, factor=0.25, min_lr=1e-05, verbose=1),
    keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
]
# model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC,
#                                            pretraining_model_manipulation_function=
#                                            get_model_with_hidden_layers_as_outputs, dropout_rate=0.5)
model_generator = AttentionBasedModelGenerator(pretrained_model_generator, OUTPUT_SPEC,
                                               pretraining_model_manipulation_function=
                                               get_model_with_hidden_layers_as_outputs, dropout_rate=0.5)
# model_generator = CNNBasedModelGenerator(pretrained_model_generator, OUTPUT_SPEC,
#                                          pretraining_model_manipulation_function=
#                                          get_model_with_hidden_layers_as_outputs, dropout_rate=0.5)
finetune(model_generator, input_encoder, OUTPUT_SPEC, train_set['seq'], train_set['label'], valid_set['seq'],
         valid_set['label'],
         seq_len=512, batch_size=32, max_epochs_per_stage=40, lr=1e-04, begin_with_frozen_pretrained_layers=True,
         lr_with_frozen_pretrained_layers=1e-02, n_final_epochs=1, final_seq_len=1024, final_lr=1e-05,
         callbacks=training_callbacks)

# 4. evaluate the model on the validation sets
test_set = valid_set
results, confusion_matrix = evaluate_by_len(model_generator, input_encoder, OUTPUT_SPEC, test_set['seq'],
                                            test_set['label'],
                                            start_seq_len=512, start_batch_size=32)
print_performance(results, confusion_matrix)
