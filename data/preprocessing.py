import re
from config.config import PLM_Config

class CAFA5_Preprocessor:

    def __init__(self, plm_name='Rostlab/prot_bert'):
        self.plm_name = plm_name.lower()
        self.valid_amino_acids = self.get_valid_amino_acids()
        self.max_seq_len = self.get_max_seq_len()

    def get_valid_amino_acids(self):
        if self.plm_name== 'rostlab/prot_bert':
            return PLM_Config.VALID_AMINO_ACIDS_PROTBERT
    def get_max_seq_len(self):
        if self.plm_name == 'rostlab/prot_bert':
            return PLM_Config.MAX_SEQ_LEN
    def format_by_PLM(self, sequence):
        if self.plm_name == 'rostlab/prot_bert':
            return self.format_for_protbert(sequence)

    def clean_sequence(self, sequence:str) -> str:
        '''
        Removes non-standard amino acids from a sequence.
        :param sequence: protein sequence
        :return:cleaned sequence in UPPERCASE.
        '''
        sequence = sequence.upper().strip()
        cleaned = ''.join([aa for aa in sequence if aa in self.valid_amino_acids])
        return cleaned

    @staticmethod
    def format_for_protbert(sequence:str) -> str:
        '''
        Format amino acid sequence for ProtBERT input.
        Adds whitespace between each residue (token).
        Example: 'ACDEF' → 'A C D E F'
        :param sequence: protein sequence
        :return: protbert formatted sequence
        '''
        return ' '.join(list(sequence.strip().upper()))

    def truncate_sequence(self, sequence: str) -> str:
        return sequence[:self.max_seq_len]

    def preprocess_sequence(self, sequence: str, clean: bool = True, truncate: bool = False) -> str:
        """
        Full preprocessing pipeline: clean + formatting regarding PLM type.
        """
        if clean:
            sequence = self.clean_sequence(sequence)

        sequence = self.format_by_PLM(sequence)
        if truncate:
            sequence = self.truncate_sequence(sequence)

        return sequence