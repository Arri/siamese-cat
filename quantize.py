import tensorflow as tf
import logging


def quantize_model(model_name, siamese_model_name, train_ds, emb_model):
    # Save the trained embedding model AFTER training
    embedding_model_name = f"embedding_{model_name}.keras"
    emb_model.save(embedding_model_name)
    print("[INFO] Training complete.")
    print(f"\tBest model saved as {siamese_model_name}.")
    logging.info(f"Best model saved as {siamese_model_name}.")
    print(f"\tThe embedding model was saved as {embedding_model_name}.")
    logging.info(f"The embedding model was saved as {embedding_model_name}.")

    # Save quantized model
    logging.info("========================= Quantization PHASE =========================")
    print("============ Quantization Phase =============")
    print("[INFO] Evaluating on test set...")

    def representative_data_gen():
        for anchor, _ in train_ds.take(100):
            yield [anchor]

    converter = tf.lite.TFLiteConverter.from_keras_model(emb_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_quant_model = converter.convert()
    quantized_model_file = f"embedding_{model_name}_int8.tflite"
    with open(quantized_model_file, "wb") as f:
        f.write(tflite_quant_model)
    logging.info("Converted the model to 8-bit. ")
    logging.info(f"Quantized model saved as {quantized_model_file}.")
    print(f"[INFO] 8-bit quantized model saved as {quantized_model_file}.")
    return quantized_model_file