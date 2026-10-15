import numpy as np


def merge_list(sample_input, full_list, key):
    if full_list[key] is None:
        full_list[key] = sample_input
    else:
        full_list[key] = np.vstack((full_list[key], sample_input))


def build_buckets(dataset_variables, split_set: str = "test"):
    print(f"\nStarting building the buckets for <{split_set}>...")
    test_video_dataset = dataset_variables[split_set]["dataset"]
    bucket_per_text = {}

    for batch, (batch_videos, batch_texts) in enumerate(test_video_dataset):
        batch_texts_decoded = [i.decode("utf-8") for i in batch_texts.numpy()]
        videos = batch_videos.numpy()

        for i, current_text in enumerate(batch_texts_decoded):
            current_video = np.expand_dims(videos[i], axis=0)
            if current_text not in bucket_per_text:
                bucket_per_text[current_text] = current_video
            else:
                bucket_per_text[current_text] = np.concatenate(
                    (bucket_per_text[current_text], current_video), axis=0)

    total_videos = 0
    for current_key in bucket_per_text.keys():
        print(f"Current bucket <{current_key}> has <{bucket_per_text[current_key].shape}> videos")
        total_videos += bucket_per_text[current_key].shape[0]
    print(f"Finished building the buckets. Total of {total_videos} videos.\n")

    return bucket_per_text
