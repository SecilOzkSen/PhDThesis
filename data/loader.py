import pandas as pd
from config.config import DatasetConfig, PLM_Config
from Bio import SeqIO


def load_sequences(model_name = "protbert") -> pd.DataFrame:
    '''
    Load protein sequences from a FASTA file into a pandas DataFrame.
    :return: Pandas dataframe with columns 'sequence_id' and 'sequence'
    '''

    records = list(SeqIO.parse(DatasetConfig.SEQ_FILE, "fasta"))
    data = []
    upper = False
    if model_name == PLM_Config.PROTBERT:
        upper = True
    for record in records:
        sequence_id = record.id
        sequence = str(record.seq).strip()
        if upper:
            sequence = sequence.upper()
        data.append((sequence_id, sequence))
    df = pd.DataFrame(data, columns=['sequence_id', 'sequence'])
    return df


def load_terms() -> pd.DataFrame:
    '''
    Load labels (GO terms) from train_terms.csv
    Expected columns are: 'sequence_id' and 'term'
    :return: Pandas dataframe where each row has a list of GO terms for a sequence.
    '''
    df = pd.read_csv(DatasetConfig.TERM_FILE, sep='\t')
    df_grouped = df.groupby('EntryID')['term'].apply(list).reset_index()
    df_grouped = df_grouped.rename(columns={'EntryID': 'sequence_id'})

    return df_grouped


def merge_sequences_and_terms(sequences: pd.DataFrame, terms: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(sequences, terms, on='sequence_id', how='inner')
    merged = merged.dropna(subset=["sequence"])
    merged = merged[merged["sequence"].str.strip() != ""]
    return merged


def load_cafa5_dataframe(model_name = "protbert") -> pd.DataFrame:
    sequences = load_sequences(model_name)
    terms = load_terms()
    merged = merge_sequences_and_terms(sequences, terms)
    return merged






