"""Khmer Medical Q&A Dataset"""

import json
import datasets

class KhmerMedicalQA(datasets.GeneratorBasedBuilder):
    """Khmer Medical Q&A Dataset."""
    
    VERSION = datasets.Version("1.0.0")
    
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "index": datasets.Value("int64"),
                "question_en": datasets.Value("string"),
                "response_en": datasets.Value("string"),
                "question_km": datasets.Value("string"),
                "response_km": datasets.Value("string"),
                "question_km_para": datasets.Value("string"),
                "response_km_para": datasets.Value("string"),
                "reasoning_summary_km": datasets.Value("string"),
                "tags": datasets.Sequence(datasets.Value("string"))
            })
        )
    
    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": "data/train-00000-of-00001.parquet"}
            )
        ]
    
    def _generate_examples(self, filepath):
        """Generate examples from Parquet file."""
        import pyarrow.parquet as pq
        
        table = pq.read_table(filepath)
        df = table.to_pandas()
        
        for idx, row in df.iterrows():
            yield idx, row.to_dict()
