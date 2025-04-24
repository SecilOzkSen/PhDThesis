import pandas as pd
from config.config import DatasetConfig
class CAFA5Loader:
    def load_sequences(self) -> pd.DataFrame:
        '''
        Load protein sequences from dataset.
        :return: Pandas dataframe with columns 'sequence_id' and 'sequence'
        '''
        df = pd.read_csv(DatasetConfig.SEQ_FILE)
        # train_sequences.csv should have both 'sequence_id' and 'sequence' columns.
        assert 'sequence_id' in df.columns and 'sequence' in df.columns, \
            "train_sequences.csv must have 'sequence_id' and 'sequence' columns"
        return df

    def load_terms(self) -> pd.DataFrame:
        '''
        Load labels (GO terms) from train_terms.csv
        Expected columns are: 'sequence_id' and 'term'
        :return: Pandas dataframe where each row has a list of GO terms for a sequence.
        '''
        df = pd.read_csv(DatasetConfig.TERM_FILE)
        assert 'sequence_id' in df.columns and 'term' in df.columns, \
            "train_terms.csv must have 'sequence_id' and 'term' columns"
        df_grouped = df.group_by('sequence_id')['term'].apply(list).reset_index()
        return df_grouped

    def merge_sequences_and_terms(self, sequences: pd.DataFrame, terms: pd.DataFrame) -> pd.DataFrame:
        merged = pd.merge(sequences, terms, on='sequence_id', how='inner')
        return merged

    def load_cafa5_dataframe(self) -> pd.DataFrame:
        sequences = self.load_sequences()
        terms = self.load_terms()
        merged = self.merge_sequences_and_terms(sequences, terms)
        return merged






