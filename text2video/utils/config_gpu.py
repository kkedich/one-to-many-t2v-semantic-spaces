import tensorflow as tf


def init_gpu(gpu_ids):
    """ Initialize the GPUs with some configurations that limit the growth of using them """

    if not isinstance(gpu_ids, list):
        gpu_ids = [gpu_ids]

    print(f"\nInitializing GPUs: {gpu_ids} ({len(gpu_ids)})")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"Physical GPUs: {gpus} ({len(gpus)})")

    if gpus:
        # Restrict TensorFlow to only use one GPU based on gpu_id
        try:
            gpus_to_set = []
            for current_gpu in gpu_ids:
                gpus_to_set.append(gpus[current_gpu])
            print(f"GPUs to set: {gpus_to_set}")

            # tf.config.set_visible_devices(gpus_to_set, "GPU")
            # Set memory growth to follow the experiment needs
            for current_gpu in gpus_to_set:
                print(f"Setting memory growth in GPU: {current_gpu}")
                tf.config.experimental.set_memory_growth(current_gpu, True)

            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


def set_multiple_gpus(nb_gpus_defined):
    # Get GPUs available
    current_gpus = tf.config.list_physical_devices("GPU")
    print("Num GPUs Available: ", len(current_gpus))

    # Set current GPUs for the experiment
    nb_gpus = len(current_gpus) if len(current_gpus) < nb_gpus_defined else nb_gpus_defined
    gpu_ids = [current_gpu for current_gpu in range(nb_gpus)]
    init_gpu(gpu_ids)
