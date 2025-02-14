import os
from gensim.models import KeyedVectors
import numpy as np
import random
from scipy.linalg import orthogonal_procrustes


class W2VModel:
    """
    A class for handling Word2Vec models stored as .kv files, with methods for
    intrinsic evaluation, normalization, vocabulary filtering, and alignment
    using orthogonal Procrustes transforms.
    """

    def __init__(self, model_path):
        """
        Initialize the W2VModel instance by loading the Word2Vec .kv file.

        Args:
            model_path (str): Path to the .kv file containing the Word2Vec model.

        Raises:
            FileNotFoundError: If the provided model_path does not exist.
            ValueError: If the file is not a valid .kv file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not model_path.endswith(".kv"):
            raise ValueError("The model file must be a .kv file.")

        self.model = KeyedVectors.load(model_path, mmap="r")
        self.vocab = set(self.model.index_to_key)
        self.vector_size = self.model.vector_size

    def evaluate(self, task, dataset_path):
        """
        Evaluate the model on a specified task (e.g., similarity or analogy).

        Args:
            task (str): The evaluation task ('similarity' or 'analogy').
            dataset_path (str): Path to the dataset file.

        Returns:
            float or dict: Evaluation results:
                - Similarity: Returns Spearman correlation as a float.
                - Analogy: Returns a dictionary of results (correct, total, accuracy).

        Raises:
            ValueError: If the task is not supported or the dataset is missing.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        if task == "similarity":
            results = self.model.evaluate_word_pairs(dataset_path)
            return results[1][0]  # Spearman correlation

        elif task == "analogy":
            results = self.model.evaluate_word_analogies(dataset_path)
            return results[0]  # Analogy accuracy

        else:
            raise ValueError("Unsupported task. Choose 'similarity' or 'analogy'.")

    def normalize(self):
        """
        Normalize vectors in the model to unit length (L2 normalization).

        Returns:
            W2VModel: The instance itself, for method chaining.
        """
        self.model.init_sims(replace=True)
        return self

    def extract_vocab(self):
        """
        Extract the model's vocabulary.

        Returns:
            set: The vocabulary of the model as a set of words.
        """
        return self.vocab

    def filter_vocab(self, reference_vocab):
        """
        Filter the model's vocabulary to include only words in the reference vocabulary.

        Args:
            reference_vocab (set): A set of words representing the reference vocabulary.

        Returns:
            W2VModel: The instance itself, for method chaining.

        Raises:
            ValueError: If the reference vocabulary is not a set.
        """
        if not isinstance(reference_vocab, set):
            raise ValueError("reference_vocab must be a set of words.")

        shared_vocab = self.vocab.intersection(reference_vocab)
        self.filtered_vectors = {word: self.model[word] for word in shared_vocab}
        self.filtered_vocab = shared_vocab
        return self

    def align_to(self, reference_model):
        """
        Align this model to a reference model using orthogonal Procrustes.

        Args:
            reference_model (W2VModel): The reference W2VModel instance to align to.

        Returns:
            W2VModel: The instance itself, for method chaining.

        Raises:
            ValueError: If the filtered vocabularies are empty or mismatched.
        """
        shared_vocab = self.filtered_vocab.intersection(reference_model.filtered_vocab)

        if not shared_vocab:
            raise ValueError("No shared vocabulary between the models.")

        # Create aligned matrices
        X = np.vstack([reference_model.filtered_vectors[word] for word in shared_vocab])
        Y = np.vstack([self.filtered_vectors[word] for word in shared_vocab])

        # Perform orthogonal Procrustes alignment
        R, _ = orthogonal_procrustes(Y, X)

        # Apply the transformation to the filtered vectors
        for word in self.filtered_vectors:
            self.filtered_vectors[word] = np.dot(self.filtered_vectors[word], R)

        return self

    def is_normalized(self, tolerance=1e-6):
        """
        Check if all word vectors in the model are L2 normalized.
    
        Args:
            tolerance (float): Allowed deviation from norm 1 due to floating-point precision.
    
        Returns:
            bool: True if all vectors are normalized, False otherwise.
        """
        norms = np.linalg.norm(self.model.vectors, axis=1)
        return np.all(np.abs(norms - 1) < tolerance)

    def is_aligned_with(self, reference_model, tolerance=1e-6):
        """
        Check if this model is already aligned with a reference model using Procrustes.
    
        Args:
            reference_model (W2VModel): The reference W2VModel instance.
            tolerance (float): Allowed deviation from identity matrix for Procrustes check.
    
        Returns:
            bool: True if the models appear to be aligned, False otherwise.
        """
        shared_vocab = self.filtered_vocab.intersection(reference_model.filtered_vocab)
    
        if not shared_vocab:
            raise ValueError("No shared vocabulary between the models to check alignment.")
    
        X = np.vstack([reference_model.filtered_vectors[word] for word in shared_vocab])
        Y = np.vstack([self.filtered_vectors[word] for word in shared_vocab])
    
        R, _ = orthogonal_procrustes(Y, X)
    
        # Check if R is approximately an identity matrix
        identity_matrix = np.eye(R.shape[0])
        return np.all(np.abs(R - identity_matrix) < tolerance)

    def evaluate_alignment(self, reference_model, tolerance=1e-3):
        """
        Evaluate the alignment quality between this model and a reference model.

        Args:
            reference_model (W2VModel): The reference W2VModel instance.
            tolerance (float): Allowed deviation from identity matrix for Procrustes check.

        Returns:
            dict: A dictionary containing various alignment diagnostics.
        """
        if not isinstance(reference_model, W2VModel):
            raise TypeError("reference_model must be an instance of W2VModel.")

        # Check normalization
        is_norm_self = self.is_normalized()
        is_norm_ref = reference_model.is_normalized()
        
        # Extract shared vocabulary
        shared_vocab = self.filtered_vocab.intersection(reference_model.filtered_vocab)
        vocab_match = self.filtered_vocab == reference_model.filtered_vocab
        
        # Prepare matrices for Procrustes
        X = np.vstack([reference_model.filtered_vectors[word] for word in shared_vocab])
        Y = np.vstack([self.filtered_vectors[word] for word in shared_vocab])

        # Compute Procrustes alignment matrix
        R, _ = orthogonal_procrustes(Y, X)
        identity_matrix = np.eye(R.shape[0])
        alignment_deviation = np.linalg.norm(R - identity_matrix)

        # Interpret deviation results
        if alignment_deviation < 1e-4:
            deviation_message = "✅ Alignment deviation is minimal. Alignment likely successful."
        elif alignment_deviation < 1e-2:
            deviation_message = "⚠️ Alignment deviation is small but nonzero. Check vocabulary consistency."
        else:
            deviation_message = "❌ Warning: Alignment deviation is significant. Possible alignment failure."

        # Final assessment
        aligned = alignment_deviation < tolerance

        # Print diagnostic information
        print("\n---------------- Normalization and Alignment Evaluation ------------------")
        print(f"Model1 normalized: {is_norm_self}")
        print(f"Model2 normalized: {is_norm_ref}")
        print(f"Shared vocabulary size: {len(shared_vocab)}")
        print(f"Filtered vocabularies match: {vocab_match}")
        print(f"Shape of X (anchor model vectors): {X.shape}")
        print(f"Shape of Y (target model vectors): {Y.shape}")
        print(f"Alignment deviation from identity: {alignment_deviation:.6f}")
        print(deviation_message)
        print(f"Models are aligned (threshold {tolerance}): {aligned}")
        print("--------------------------------------------------------------------------\n")

        # Return detailed results as a dictionary
        return {
            "is_normalized_self": is_norm_self,
            "is_normalized_ref": is_norm_ref,
            "shared_vocab_size": len(shared_vocab),
            "vocab_match": vocab_match,
            "matrix_shape_X": X.shape,
            "matrix_shape_Y": Y.shape,
            "alignment_deviation": alignment_deviation,
            "alignment_message": deviation_message,
            "is_aligned": aligned
        }

    def cosine_similarity(self, word1, word2):
        """
        Compute the cosine similarity between two words in the model.
    
        Args:
            word1 (str): The first word.
            word2 (str): The second word.
    
        Returns:
            float: Cosine similarity score between the two words.
    
        Raises:
            KeyError: If either word is not in the vocabulary.
        """
        if word1 not in self.vocab or word2 not in self.vocab:
            raise KeyError(f"One or both words ('{word1}', '{word2}') are not in the vocabulary.")
    
        return self.model.similarity(word1, word2)

    def cosine_similarity_mean(self, reference_model):
        """
        Compute the mean cosine similarity with a reference model across shared words.
        
        Args:
            reference_model (W2VModel): The anchor model (reference).
        
        Returns:
            float: The mean cosine similarity of shared words, or None if no shared words exist.
        """
        common_words = self.vocab.intersection(reference_model.vocab)
        
        if not common_words:
            print("⚠️ Warning: No shared words between models.")
            return None
    
        similarities = [
            np.dot(self.model[word], reference_model.model[word]) for word in common_words
        ]
    
        return np.mean(similarities)

    def mean_cosine_similarity_to_all(self, word, excluded_words=None):
        """
        Compute the mean cosine similarity of a given word with every other word in the vocabulary.
    
        Args:
            word (str): The word for which to compute the mean similarity.
    
        Returns:
            float: Mean cosine similarity score of the word with all other words in the vocabulary.
    
        Raises:
            KeyError: If the word is not in the vocabulary.
        """
        if word not in self.vocab:
            raise KeyError(f"Word '{word}' is not in the vocabulary.")

        total_similarity = 0
        count = 0
    
        for other_word in self.vocab:
            if other_word == word or other_word in excluded_words:
                continue  # Skip self-similarity
            total_similarity += self.cosine_similarity(word, other_word)
            count += 1
    
        return total_similarity / count if count > 0 else 0

    def compute_weat(self, targ1, targ2, attr1, attr2, num_permutations=10000, return_std=False):
        """
        Compute WEAT effect size, p-value, and optionally return standard deviation from permutations.
        Fully follows Caliskan et al.'s method.
        """
        missing_words = [word for word in (targ1 + targ2 + attr1 + attr2) if word not in self.vocab]
        if missing_words:
            print(f"⚠️ Warning: The following words are missing from the model and will be ignored: {missing_words}")
    
        def cosine_similarity_list(words1, words2):
            return [self.model.similarity(w1, w2) for w1 in words1 for w2 in words2 if w1 in self.vocab and w2 in self.vocab]
    
        # Compute original association scores
        S_targ1_attr1 = cosine_similarity_list(targ1, attr1)
        S_targ1_attr2 = cosine_similarity_list(targ1, attr2)
        S_targ2_attr1 = cosine_similarity_list(targ2, attr1)
        S_targ2_attr2 = cosine_similarity_list(targ2, attr2)
    
        # Compute test statistic (difference in means)
        mean_diff_targ1 = np.mean(S_targ1_attr1) - np.mean(S_targ1_attr2)
        mean_diff_targ2 = np.mean(S_targ2_attr1) - np.mean(S_targ2_attr2)
    
        # Compute pooled standard deviation across X ∪ Y (as in Caliskan et al.)
        all_similarities = S_targ1_attr1 + S_targ1_attr2 + S_targ2_attr1 + S_targ2_attr2
        pooled_std = np.std(all_similarities, ddof=1)
    
        if pooled_std == 0:
            print("⚠️ Warning: No variation in similarities. Returning NaN for WEAT effect size.")
            return (np.nan, None, None) if return_std else (np.nan, None)
    
        # Compute observed WEAT effect size
        weat_effect_size = (mean_diff_targ1 - mean_diff_targ2) / pooled_std
    
        if num_permutations == 0:
            return (weat_effect_size, None, pooled_std) if return_std else (weat_effect_size, None)
    
        # Permutation Test (Shuffle only target words `X` and `Y`)
        combined_targets = targ1 + targ2
        n = len(targ1)
        permuted_effect_sizes = []
    
        for _ in range(num_permutations):
            # Shuffle only the target words (not attributes)
            perm_targ1 = random.sample(combined_targets, n)
            perm_targ2 = [w for w in combined_targets if w not in perm_targ1]
    
            S_perm_targ1_attr1 = cosine_similarity_list(perm_targ1, attr1)
            S_perm_targ1_attr2 = cosine_similarity_list(perm_targ1, attr2)
            S_perm_targ2_attr1 = cosine_similarity_list(perm_targ2, attr1)
            S_perm_targ2_attr2 = cosine_similarity_list(perm_targ2, attr2)
    
            perm_mean_diff_targ1 = np.mean(S_perm_targ1_attr1) - np.mean(S_perm_targ1_attr2)
            perm_mean_diff_targ2 = np.mean(S_perm_targ2_attr1) - np.mean(S_perm_targ2_attr2)
    
            # Compute effect size for this permutation using the original pooled std dev
            perm_effect_size = (perm_mean_diff_targ1 - perm_mean_diff_targ2) / pooled_std
            permuted_effect_sizes.append(perm_effect_size)
    
        # Compute p-value
        p_value = np.mean(np.array(permuted_effect_sizes) >= weat_effect_size)
    
        # Compute standard deviation of the permuted effect sizes (confidence interval estimate)
        std_dev = np.std(permuted_effect_sizes, ddof=1) if return_std else None
    
        return (weat_effect_size, p_value, std_dev) if return_std else (weat_effect_size, p_value)

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

        aligned_model = KeyedVectors(vector_size=self.vector_size)
        aligned_model.add_vectors(
            list(self.filtered_vectors.keys()), list(self.filtered_vectors.values())
        )
        aligned_model.save(output_path)