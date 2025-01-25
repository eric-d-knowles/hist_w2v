import logging
import os
from gensim.models import KeyedVectors
import numpy as np
from scipy.linalg import orthogonal_procrustes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class W2VModel:
    """
    A class for handling Word2Vec models stored as .kv files, with methods for
    instrinsic evaluation, normalization, vocabulary filtering, alignment using
    orthogonal Procrustes.
    """

    def __init__(self, model_path):
        """
        Initialize the W2VModel instance by loading the Word2Vec .kv file.

        Args:
            model_path (str): Path to the .kv file containing the Word2Vec
            model.

        Raises:
            FileNotFoundError: If the provided model_path does not exist.
            ValueError: If the file is not a valid .kv file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not model_path.endswith(".kv"):
            raise ValueError("The model file must be a .kv file.")

        logger.info(f"Loading model from {model_path}")
        self.model = KeyedVectors.load(model_path, mmap="r")
        self.vocab = set(self.model.index_to_key)
        self.vector_size = self.model.vector_size

    def evaluate(self, task, dataset_path):
        """
        Evaluate the model on a specified task (e.g., similarity or analogy).

        Args:
            task (str): The evaluation task ('similarity' or 'analogy').
            dataset_path (str): Path to the dataset file (e.g., WordSim-353
                                for similarity or Google's analogy dataset).

        Returns:
            float or dict: Evaluation results:
                - Similarity: Returns Spearman correlation as a float.
                - Analogy: Returns a dictionary of results (correct, total,
                  accuracy).

        Raises:
            ValueError: If the task is not supported or the dataset is missing.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        if task == "similarity":
            logger.info(
                f"Evaluating model on similarity dataset: {dataset_path}"
            )
            results = self.model.evaluate_word_pairs(dataset_path)
            spearman_correlation = results[1][0]
            logger.info(f"Spearman correlation: {spearman_correlation:.4f}")
            return spearman_correlation

        elif task == "analogy":
            logger.info(f"Evaluating model on analogy dataset: {dataset_path}")
            accuracy = self.model.evaluate_word_analogies(dataset_path)
            logger.info(
                f"Analogy accuracy: {accuracy['correct']}/{accuracy['total']} "
                f"({accuracy['correct'] / accuracy['total']:.4%})"
            )
            return accuracy

        else:
            raise ValueError(
                "Unsupported task. Choose 'similarity' or 'analogy'."
            )

    def normalize(self):
        """
        Normalize vectors in the model to unit length (L2 normalization).

        Returns:
            W2VModel: The instance itself, for method chaining.
        """
        logger.info("Normalizing vectors to unit length.")
        self.model.init_sims(replace=True)
        return self

    def extract_vocab_as_reference(self):
        """
        Extract the model's vocabulary to be used as a reference vocabulary.

        Returns:
            set: The vocabulary of the model as a set of words.
        """
        logger.info("Extracting vocabulary as reference.")
        return self.vocab

    def filter_vocab(self, reference_vocab):
        """
        Filter the model's vocabulary to include only words in the reference
        vocabulary.

        Args:
            reference_vocab (set): A set of words representing the reference
            vocabulary.

        Returns:
            W2VModel: The instance itself, for method chaining.

        Raises:
            ValueError: If the reference vocabulary is not a set.
        """
        if not isinstance(reference_vocab, set):
            raise ValueError("reference_vocab must be a set of words.")

        logger.info("Filtering vocabulary based on reference vocabulary.")
        shared_vocab = self.vocab.intersection(reference_vocab)
        self.filtered_vectors = {
            word: self.model[word] for word in shared_vocab
        }
        self.filtered_vocab = shared_vocab
        logger.info(f"Filtered vocabulary size: {len(self.filtered_vocab)}")
        return self

    def align_to(self, reference_model):
        """
        Align this model to a reference model using orthogonal Procrustes.

        Args:
            reference_model (W2VModel): The reference W2VModel instance to
            align to.

        Returns:
            W2VModel: The instance itself, for method chaining.

        Raises:
            ValueError: If the filtered vocabularies are empty or mismatched.
        """
        shared_vocab = self.filtered_vocab.intersection(
            reference_model.filtered_vocab
        )

        if not shared_vocab:
            raise ValueError("No shared vocabulary between the models.")

        logger.info(
            "Aligning model to reference model using orthogonal Procrustes."
        )

        # Create aligned matrices
        X = np.vstack([
            reference_model.filtered_vectors[word] for word in shared_vocab
        ])
        Y = np.vstack([
            self.filtered_vectors[word] for word in shared_vocab
        ])

        # Perform orthogonal Procrustes alignment
        R, _ = orthogonal_procrustes(Y, X)

        # Apply the transformation to the filtered vectors
        for word in self.filtered_vectors:
            self.filtered_vectors[word] = np.dot(
                self.filtered_vectors[word], R
            )

        logger.info("Alignment complete.")
        return self

    def save(self, output_path):
        """
        Save the filtered and aligned model to the specified path.

        Args:
            output_path (str): Path to save the aligned .kv model.

        Raises:
            ValueError: If no filtered vectors are available to save.
        """
        if not hasattr(self, "filtered_vectors") or not self.filtered_vectors:
            raise ValueError("No filtered vectors available to save.")

        logger.info(f"Saving aligned model to {output_path}")
        aligned_model = KeyedVectors(vector_size=self.vector_size)
        aligned_model.add_vectors(
            list(self.filtered_vectors.keys()),
            list(self.filtered_vectors.values())
        )
        aligned_model.save(output_path)
        logger.info("Model saved successfully.")