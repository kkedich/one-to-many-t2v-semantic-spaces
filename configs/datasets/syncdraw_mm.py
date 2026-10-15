cfg = dict(
    dataset_name="syncdraw_mm",
    input=dict(
        channels=1
    ),
    bucket_info="./data/syncdraw_mm/records_syncdraw_mm_with_text/buckets/train.json",
    one_video_per_text="./data/syncdraw_mm/records_syncdraw_mm_with_text/buckets/train/train_one_video_per_text.record",
    records=dict(
        root_wo_text="./data/syncdraw_mm/records/",
        root="./data/syncdraw_mm/records_syncdraw_mm_with_text/",
        train="train_record_tfrecord_files.txt",
        val="val_record_tfrecord_files.txt",
        test="test_record_tfrecord_files.txt",
    )
)
