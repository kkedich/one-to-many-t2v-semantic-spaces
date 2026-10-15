import pandas as pd
from pathlib import Path


def dict_to_markdown_table(result_per_metric: dict):
    df = pd.DataFrame(result_per_metric.items(), columns=['Metric', 'Value'])
    df['Value'] = df['Value'].map('{:,.3f}'.format)
    markdown_table = df.to_markdown(index=False)
    print(markdown_table)

    return markdown_table


def images_to_markdown_strings(images: list):
    markdown_string_samples = ""
    for image_path in images:
        markdown_string_samples += f"<img src={Path(image_path).name}>\n"

    return markdown_string_samples
