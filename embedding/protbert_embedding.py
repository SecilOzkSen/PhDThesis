import torch
from transformers import BertModel, BertTokenizer
from data.preprocessing import CAFA5_Preprocessor
from config.config import PLM_Config
from typing import List
from tqdm import tqdm
class ProtBERTEmbedder:
    def __init__(self,
                 model_name: str = 'Rostlab/prot_bert',
                 model_path: str = None,
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        if model_path is None:
            self.model = BertModel.from_pretrained(model_name, force_download=True).to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        else:
            self.model = BertModel.from_pretrained(model_path, force_download=True).to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
        self.cafa5_preprocessor = CAFA5_Preprocessor(plm_name=model_name)
        self.model.eval()

    def get_embedding(self, sequence:str) -> torch.Tensor: # for single protein
        '''
        Returns the mean pooled embedding for a single protein sequence
        '''
        formatted_sequence = self.cafa5_preprocessor.preprocess_sequence(sequence)
        tokens = self.tokenizer(formatted_sequence,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=PLM_Config.MAX_SEQ_LEN_PROTBERT)
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = outputs.last_hidden_state.mean(dim=1) # we are not doing inference we are using embeddings only!
        return embedding.squeeze.cpu()

    def get_embeddings(self, sequences: List[str], batch_size:int = 128) -> List[torch.Tensor]:
        embeddings = []
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seqs = sequences[i:i+batch_size]
            formatted = [self.cafa5_preprocessor.preprocess_sequence(seq) for seq in batch_seqs]

            tokens = self.tokenizer(formatted,
                                    return_tensors='pt',
                                    padding=True,
                                    truncation=True,
                                    max_length=PLM_Config.MAX_SEQ_LEN_PROTBERT)
            input_ids = tokens['input_ids'].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pooled = outputs.last_hidden_state.mean(dim = 1) # mean pooling the embeddings.
            embeddings.extend(pooled.cpu())

        return embeddings


