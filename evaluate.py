from tensorflow import keras

from proteinbert import load_pretrained_model, finetune, evaluate_by_len
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from util.load_dataset import load_dataset, OUTPUT_SPEC, print_performance
from util.ensemble_predict import multi_classifier
from model import AttentionBasedModelGenerator, CNNBasedModelGenerator

# Evaluate on the independent test set MXD494
train_set = load_dataset("myDatasets/mxd494/training_set.fasta")
valid_set = load_dataset("myDatasets/validation_set.fasta")
ldr_train_set = load_dataset("myDatasets/mxd494/ldr_training_set.fasta")
ldr_valid_set = load_dataset("myDatasets/ldr_validation_set.fasta")
sdr_train_set = load_dataset("myDatasets/mxd494/sdr_training_set.fasta")
sdr_valid_set = load_dataset("myDatasets/sdr_validation_set.fasta")
test_set = valid_set

# 2.loading the pre-trained model
pretrained_model_generator, input_encoder = load_pretrained_model()

# 3. Training
model_save_path = "models/testset_experiment/mxd494/GDRs.h5"
training_callbacks = [
    keras.callbacks.ReduceLROnPlateau(patience=1, factor=0.25, min_lr=1e-05, verbose=1),
    keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
]

model_generator = AttentionBasedModelGenerator(pretrained_model_generator, OUTPUT_SPEC,
                                               pretraining_model_manipulation_function=
                                               get_model_with_hidden_layers_as_outputs, dropout_rate=0.5)
ldr_model_generator = AttentionBasedModelGenerator(pretrained_model_generator, OUTPUT_SPEC,
                                                   pretraining_model_manipulation_function=
                                                   get_model_with_hidden_layers_as_outputs, dropout_rate=0.5)
sdr_model_generator = AttentionBasedModelGenerator(pretrained_model_generator, OUTPUT_SPEC,
                                                   pretraining_model_manipulation_function=
                                                   get_model_with_hidden_layers_as_outputs, dropout_rate=0.5)

finetune(model_generator, input_encoder, OUTPUT_SPEC, train_set['seq'], train_set['label'], valid_set['seq'],
         valid_set['label'],
         seq_len=512, batch_size=32, max_epochs_per_stage=40, lr=1e-04, begin_with_frozen_pretrained_layers=True,
         lr_with_frozen_pretrained_layers=1e-02, n_final_epochs=1, final_seq_len=1024, final_lr=1e-05,
         callbacks=training_callbacks)
finetune(ldr_model_generator, input_encoder, OUTPUT_SPEC, ldr_train_set['seq'], ldr_train_set['label'],
         ldr_valid_set['seq'],
         ldr_valid_set['label'],
         seq_len=512, batch_size=32, max_epochs_per_stage=40, lr=1e-04, begin_with_frozen_pretrained_layers=True,
         lr_with_frozen_pretrained_layers=1e-02, n_final_epochs=1, final_seq_len=1024, final_lr=1e-05,
         callbacks=training_callbacks)
finetune(sdr_model_generator, input_encoder, OUTPUT_SPEC, sdr_train_set['seq'], sdr_train_set['label'],
         sdr_valid_set['seq'],
         sdr_valid_set['label'],
         seq_len=512, batch_size=32, max_epochs_per_stage=40, lr=1e-04, begin_with_frozen_pretrained_layers=True,
         lr_with_frozen_pretrained_layers=1e-02, n_final_epochs=1, final_seq_len=1024, final_lr=1e-05,
         callbacks=training_callbacks)

# 4. Evaluating the performance on the validation sets
# results, confusion_matrix = evaluate_by_len(sdr_model_generator, input_encoder, OUTPUT_SPEC, test_set['seq'],
#                                             test_set['label'],
#                                             start_seq_len=512, start_batch_size=32)
# print_performance(results, confusion_matrix)
model_generators = [model_generator,ldr_model_generator,sdr_model_generator]
loaded_weights = []
results, confusion_matrix = multi_classifier(model_generators, input_encoder, OUTPUT_SPEC, test_set['seq'],
                                            test_set['label'], loaded_weights=loaded_weights,
                                            start_seq_len=512, start_batch_size=32)