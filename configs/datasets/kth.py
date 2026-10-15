cfg = dict(
    dataset_name="kth",
    input=dict(
        channels=1
    ),
    bucket_info=None,
    records=dict(
        root_wo_text="./data/kth/records_new/",
        root="./data/kth/records_new_augmented_with_text/",
        train="train_record_tfrecord_files.txt",
        val="val_record_tfrecord_files.txt",
        test="test_record_tfrecord_files.txt",
    )
)
