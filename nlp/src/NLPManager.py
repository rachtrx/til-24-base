# from typing import Dict
# from transformers import pipeline

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict

class NLPManager:
    def __init__(self, model_name="./models/nlp_4"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nlp = spacy.load("en_core_web_sm")
        self.descriptive_vector = self.get_mean_embedding_spacy("colorful vibrant bright vivid") # TODO since data is mostly color describing targets
        self.weapon_vector = self.get_mean_embedding_spacy("gun rifle knife bomb missile grenade weapon")
    
    @staticmethod
    def get_mean_embedding_spacy(text):
        doc = self.nlp(text)
        return np.mean([token.vector for token in doc if token.has_vector], axis=0)
    
    @staticmethod
    def clean_transcript(transcript):
        # Convert to lowercase
        transcript = transcript.lower()

        # Remove punctuation except hyphens
        cleaned_transcript = re.sub(r'[^\w\s-]', '', transcript)

        # Remove extra whitespace
        cleaned_transcript = re.sub(r'\s+', ' ', cleaned_transcript).strip()

        return cleaned_transcript

    @staticmethod
    def word_to_num(word):
        """Convert number words to digits if possible, and handle variations."""
        word = word.replace(',', '')  # Remove commas
        num_dict = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
        }

        # Regular expression patterns for numbers
        pattern_actual = r'^(zero|one|two|three|four|five|six|seven|eight|nine)$'
        pattern_variations = rf'^{pattern_actual}(er|s)?$'

        match = re.match(pattern_variations, word)

        if match:
            actual = match.group(1)
            return num_dict[actual]

        return word

    @staticmethod
    def select_best_entity(entities, representative_vector_fn, concept_vector):
        best_entity = None
        best_similarity = -1
        for entity in entities:
            entity_vector = representative_vector_fn(entity)
            similarity = cosine_similarity([entity_vector], [concept_vector])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_entity = entity
        return best_entity
    
    def predict_context(context, model):
        sentence = self.clean_transcript(context)
        tokens = sentence.split()
        tokens = [self.word_to_num(word) for word in tokens]
        inputs = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding="max_length", truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = outputs.logits.cpu().numpy()
        preds = np.argmax(predictions, axis=2)
        word_ids = inputs.word_ids(batch_index=0)

        pred_labels = [id2label[pred] if word_idx is not None else 'O' for pred, word_idx in zip(preds[0], word_ids)]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        entity_list = []
        entity_types = {}
        current_entity = []
        current_label = None

        for token, label in zip(tokens, pred_labels):
            if label.startswith("B-"):
                if current_entity and current_label:
                    if current_label not in entity_types:
                        entity_types[current_label] = []
                    entity_types[current_label].append(" ".join(current_entity))
                current_entity = [token]
                current_label = label[2:]
            elif label.startswith("I-") and current_label == label[2:]:
                current_entity.append(token)
            else:
                if current_entity and current_label:
                    if current_label not in entity_types:
                        entity_types[current_label] = []
                    entity_types[current_label].append(" ".join(current_entity))
                current_entity = []
                current_label = None

        if current_entity and current_label:
            if current_label not in entity_types:
                entity_types[current_label] = []
            entity_types[current_label].append(" ".join(current_entity))
        
        result = {"heading": None, "target": None, "tool": None}
        
        headings = entity_types.get("HEADING")
        if headings:
            result['heading'] = headings[0]
        targets = entity_types.get("TARGET")
        if targets and len(targets) > 0:
            result['target'] = sorted(entities, key=lambda x: cosine_sim(self.get_mean_embedding_spacy(x), self.descriptive_vector), reverse=True)[0]
        tools = entity_types.get("TOOL")
        if tools and len(tools) > 0:
            result['tool'] = sorted(entities, key=lambda x: cosine_sim(self.get_mean_embedding_spacy(x), self.weapon_vector), reverse=True)[0]

        return result
    
    def qa(self, context: str) -> Dict[str, str]:
        return self.predict_context(context)
