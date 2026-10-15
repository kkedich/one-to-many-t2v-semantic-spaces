import tensorflow as tf
from pathlib import Path

from text2video.utils import dbpedia_utils
from text2video.utils.config import Config
from text2video.utils.dbpedia_utils import pre_process_sentence
from text2video.utils.tensor_utils import check_nan


def instantiate_text_model(cfg_text_experiment: Config, model_type: str = "base", offline_models: bool = True):

    print(f"\nLoading text model <{model_type.upper()}>...")

    # Build text additional requirements
    word_dict = dbpedia_utils.build_word_dictionary(
        cfg_text_experiment.dataset_cfg.original_train_file,
        output_file=cfg_text_experiment.dataset_cfg.word_dict,
    )
    print(f"{len(word_dict)} words in dictionary")

    if model_type == "base":
        # Load the text model
        print(f"Loading text model from <{cfg_text_experiment.output_dir}>")
        path_text_model = Path(cfg_text_experiment.output_dir)
        if not path_text_model.exists():
            raise FileNotFoundError(f"Directory <{path_text_model}> does not exist!")
        text_encoder_model = tf.keras.models.load_model(
            f"{cfg_text_experiment.output_dir}/encoder_model")
        text_encoder_model.summary()

        text_models = {
            "encoder": text_encoder_model,
            "word_dict": word_dict
        }

    elif model_type == "clip":
        from transformers import AutoTokenizer, CLIPModel

        if offline_models:
            model_path = "/home/kbogdan/saved_models/clip-vit-base-patch32"

            model = CLIPModel.from_pretrained(model_path, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        else:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        text_models = {
            "encoder": model,
            "tokenizer": tokenizer,
            "word_dict": word_dict
        }
    else:
        raise ValueError(f"Not valid option: {model_type}")

    return text_models


def get_text_embeddings(batch_texts_decoded, models, model_type):
    text_encoder_model = models["text_encoder_model"]

    if model_type == "base":
        sentence_in_tokens = pre_process_sentence(
            batch_texts_decoded, word_dict=models["word_dict"])
        input_tokens = tf.cast(sentence_in_tokens, dtype=tf.int64)

        text_encoder_embeddings, _, _ = text_encoder_model(input_tokens, training=False)
        check_nan(text_encoder_embeddings, tensor_name="text_encoder_embeddings")
    else:
        # CLIP
        tokenizer = models["tokenizer"]
        input_tokens = tokenizer(batch_texts_decoded, padding=True, return_tensors="pt")

        text_encoder_embeddings = text_encoder_model.get_text_features(
            **input_tokens).detach().numpy()
        text_encoder_embeddings = tf.cast(text_encoder_embeddings, dtype=tf.float32)

    return text_encoder_embeddings
