cfg = dict(
    dataset_name="tacos",
    input=dict(
        channels=3
    ),
    bucket_info=None,
    records=dict(
        root_wo_text="./data/tacos/dataset/records_new/",
        root="./data/tacos/dataset/records_new/",
        train="train_record_tfrecord_files.txt",
        val="val_record_tfrecord_files.txt",
        test="test_record_tfrecord_files.txt",
    ),
    # Require dict to consider as part of Tacos dataset, for other datasets
    # such as SyncDraw and KTH, this is already hard-coded
    required_dicts=[
        "./data/tacos/dataset/dictionary_words.json"
    ]
)
