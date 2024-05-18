from typing import Dict
from transformers import pipeline

class NLPManager:
    def __init__(self, model_name="path/to/your/trained/model"):
        self.nlp_pipeline = pipeline("ner", model=model_name)

    def qa(self, context: str) -> Dict[str, str]:
        ner_results = self.nlp_pipeline(transcript)
        target = None
        heading = None
        tool = None

        for entity in ner_results:
            if entity['entity'] == 'B-TARGET':
                target = entity['word']
            elif entity['entity'] == 'B-HEADING':
                heading = entity['word']
            elif entity['entity'] == 'B-TOOL':
                tool = entity['word']

        return {"heading": heading, "target": target, "tool": tool}
