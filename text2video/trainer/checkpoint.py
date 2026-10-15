import tensorflow as tf
from pathlib import Path


def load_weights_from(pre_trained_model, target_model):
    print(f"Loading target model <{target_model.name}>...")
    layer_dict_pre_trained_model = dict([(layer.name, layer) for layer in pre_trained_model.layers])

    for i, layer in enumerate(target_model.layers):
        if len(layer.weights) > 0:
            # Layer that have weights, a layer without weights has [] as value
            print(
                f"Layer: {layer.name} loaded from pre_trained_model: {layer_dict_pre_trained_model[layer.name]}")
            pre_trained_weights = layer_dict_pre_trained_model[layer.name].get_weights()
            target_model.get_layer(layer.name).set_weights(pre_trained_weights)
    print(f"Done <{target_model.name}>.")


def load_unique_layer(pre_trained_model, target_model, target_layer_name: str):
    print(f"Loading unique target layer <{target_layer_name}>...")
    layer_dict_pre_trained_model = dict([(layer.name, layer) for layer in pre_trained_model.layers])

    target_layer = None
    for i, layer in enumerate(target_model.layers):
        # Ignore input shape layers
        if target_layer_name not in layer.name or "input" in layer.name:
            continue
        target_layer = layer
        break

    pre_trained_weights = layer_dict_pre_trained_model[target_layer_name].get_weights()
    target_model.get_layer(target_layer.name).set_weights(pre_trained_weights)
    print(f"Done <{target_layer_name}>.")


def restore_model(ckpt_path, video_generator, new_load_approach=True):
    if not Path(ckpt_path).exists():
        raise FileExistsError(f"File <{ckpt_path}> does not exist!")

    if new_load_approach:
        ckpt_generator = tf.train.Checkpoint(model=video_generator)
        gen_manager = tf.train.CheckpointManager(
            ckpt_generator,
            directory=ckpt_path,
            max_to_keep=3,
            checkpoint_name="ckpt_video_generator_"
        )

        # Load checkpoints
        latest_gen_ckp = gen_manager.latest_checkpoint
        if latest_gen_ckp is not None:
            print(f"Restoring from <{latest_gen_ckp}>")
            status = ckpt_generator.restore(latest_gen_ckp)
            status.expect_partial()
        else:
            raise ValueError(f"Latest checkpoint is not valid! <{latest_gen_ckp}>")
    else:
        # Loading the models as in the previous approach
        print("Loading the model with <tf.keras.models.load_model> function...")
        video_generator = tf.keras.models.load_model(ckpt_path)

    print("Model restored!\n")

    return video_generator


def video_ae_checkpoint_restore(
        last_epoch: int,
        gen_manager: tf.train.CheckpointManager,
        disc_manager: tf.train.CheckpointManager,
        ckpt_generator: tf.train.Checkpoint,
        ckpt_disc: tf.train.Checkpoint
):
    # Load checkpoints
    latest_gen_ckp = gen_manager.latest_checkpoint
    if latest_gen_ckp is None:
        raise ValueError(f"Latest checkpoint is None. Please check: <{latest_gen_ckp}>")

    print(f"Restoring from <{latest_gen_ckp}>")
    status = ckpt_generator.restore(latest_gen_ckp)
    status.expect_partial()

    if disc_manager is not None:
        latest_disc_ckp = disc_manager.latest_checkpoint
        print(f"Restoring from <{latest_disc_ckp}>")
        status = ckpt_disc.restore(latest_disc_ckp)
        status.expect_partial()

    if last_epoch is not None:
        try:
            if last_epoch == 0:
                last_epoch = int(ckpt_generator.step.values[0]) + 1
        except Exception as error:
            print(error)
    print(
        f"Resume training from epoch <{last_epoch}> (args.last_epoch={last_epoch})...")

    return last_epoch


def load_checkpoint(
        ckpt_manager: tf.train.CheckpointManager,
        ckpt_models: tf.train.Checkpoint,
        last_epoch_input: int = 0
):
    # Load checkpoints
    latest_ckp = ckpt_manager.latest_checkpoint
    if latest_ckp is not None:
        print(f"Restoring from <{latest_ckp}>")
        status = ckpt_models.restore(latest_ckp)
        status.expect_partial()

        try:
            last_epoch = last_epoch_input
            if last_epoch_input == 0:
                print(ckpt_models.step)
                print(f"Current epoch in checkpoint: {ckpt_models.step.numpy()}")
                last_epoch = int(ckpt_models.step.numpy()) + 1
        except Exception as error:
            print(error)
            raise ValueError("Something is wrong when loading the last valid epoch")
        print(
            f"Resumed training from epoch <{last_epoch}> (args.last_epoch={last_epoch_input})\n")
    else:
        raise ValueError(
            "Latest checkpoint is None, no checkpoint found. Check the checkpoint dir")

    return last_epoch
