"""
===============================================================================================
Author:      [Pratyay Dutta]
Affiliation: [VISLab (Bir Bhanu)/Computer Science/University or California, Riverside]
===============================================================================================

README: Three-Way Multiple Sequence Alignment (MSA) with Divide-and-Conquer and Numba Acceleration
===============================================================================================

-----------------------------------------------------------------------------------------------
REQUIRED PACKAGES
-----------------------------------------------------------------------------------------------
- numpy
- numba
- biopython
- psutil

Install with:
    pip install numpy numba biopython psutil

-----------------------------------------------------------------------------------------------
HOW TO USE
-----------------------------------------------------------------------------------------------

1. **Edit the Input DNA Sequences**
   - Modify the three variables `raw_input1`, `raw_input2`, and `raw_input3` near the top of the `main()` function.
   - Each should be a string representing a DNA sequence (A, C, G, T).

2. **Run the Script**
   - From the terminal or your IDE, run: 
        python <scriptname>.py

   - This will:
       - Write the three input DNA sequences to separate `.txt` files
       - Combine them into a FASTA file (`example.fasta`)
       - Perform three-way MSA using the divide-and-conquer algorithm

3. **View the Output**
   - The script will print alignment statistics (score, time, memory, matches, length)
   - The aligned sequences will be printed in the console.

-----------------------------------------------------------------------------------------------
FUNCTION ORGANIZATION
-----------------------------------------------------------------------------------------------

- **Helper Functions:**
    - `dnagen(raw_text, output_file)`: Cleans and saves a raw DNA sequence to a file.
    - `fastagen(seq_files, output_fasta)`: Combines three sequence files into a single FASTA file.
    - `blast_sigma(a, b, c)`: Example scoring function for three residues.
    - Memory monitoring utilities.

- **Numba-Accelerated Core:**
    - `build_score_array`, `letter_to_int_seq`, `int_seq_to_string`: Utilities for encoding/decoding.
    - `numba_base3d_align`, `numba_forward3dslice`, `numba_backward3dslice`: 
      High-performance functions for full DP and DC slices.

- **Aligner Class:**
    - Encapsulates all major MSA logic.
    - Key methods:
        - `BASE3DALIGN`: Full 3D DP with traceback.
        - `FORWARD3DSLICE` / `BACKWARD3DSLICE`: Compute slices for divide-and-conquer.
        - `DC_THREEWAY_ALIGN`: Recursive divide-and-conquer alignment.
        - `run_from_fasta`: Loads input, runs alignment, prints results.

-----------------------------------------------------------------------------------------------
NOTES
-----------------------------------------------------------------------------------------------

- The script creates intermediate files (`test1.txt`, `test2.txt`, `test3.txt`, `example.fasta`).
- You may adapt scoring, threshold, or input/output logic as needed for your application.

===============================================================================================
"""




# -----------------------------------------------------------------------------------------------------------
# -------------------------------------- Imports ------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

from typing import Callable, Tuple
from Bio import SeqIO
import time
import numpy as np
from numba import njit
import psutil
import threading
import os

# -----------------------------------------------------------------------------------------------------------
# -------------------------------------- Helper Functions ---------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

def dnagen(raw_text: str, output_file: str = "sequence.txt"):
    """
    Convert a block of raw DNA text (possibly with line numbers) into a continuous sequence file.

    Parameters:
        raw_text (str): Multiline string containing DNA sequence, optionally with line numbers.
        output_file (str): Output filename for the processed DNA sequence.

    Writes:
        output_file: Contains the DNA sequence as a single string (lowercase).
    """
    lines = raw_text.strip().splitlines()
    dna_sequence = ''
    for line in lines:
        parts = line.strip().split()
        if parts and parts[0].isdigit():
            parts = parts[1:]  # Remove line number if present
        dna_sequence += ''.join(parts)
    dna_sequence = dna_sequence.lower()
    with open(output_file, "w") as f:
        f.write(dna_sequence)
    print(f"DNA sequence saved to '{output_file}'.")


def fastagen(seq_files, output_fasta="example.fasta"):
    """
    Combine three DNA sequence files into a single FASTA file for MSA.

    Parameters:
        seq_files (list): List of three .txt files, each containing a DNA sequence.
        output_fasta (str): Name of the output FASTA file.

    Writes:
        output_fasta: FASTA file with three entries, labeled >seq1, >seq2, >seq3.
    """
    if len(seq_files) != 3:
        raise ValueError("Please provide exactly three .txt files.")
    with open(output_fasta, "w") as fasta:
        for i, filepath in enumerate(seq_files, start=1):
            with open(filepath, "r") as f:
                sequence = f.read().strip().replace('\n', '')
                fasta.write(f">seq{i}\n{sequence}\n")
    print(f"FASTA file '{output_fasta}' created from input .txt files.")


def blast_sigma(a, b, c):
    """
    Simple DNA sum-of-pairs scoring function used for MSA.

    Parameters:
        a, b, c (str): Single-character residues ('A', 'C', 'G', 'T', or '-').

    Returns:
        int: Sum of pairwise scores between a, b, and c.
    """
    def score(x, y):
        if x == '-' and y == '-':
            return 0
        elif x == '-' or y == '-':
            return -8
        elif x == y:
            return 5
        else:
            return -4
    return score(a, b) + score(a, c) + score(b, c)


def monitor_peak_memory(stop_event, peak_rss):
    """
    Monitor and record the peak RSS memory usage of the process.

    Parameters:
        stop_event (threading.Event): Event to signal when to stop monitoring.
        peak_rss (list): Single-item list to store the peak RSS observed (in bytes).

    Continuously updates peak_rss[0] until stop_event is set.
    """
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        rss = process.memory_info().rss
        peak_rss[0] = max(peak_rss[0], rss)
        time.sleep(0.01)  # check every 10 ms



# ----------------------------------------------------------------------------------------------------------
# --------------------------------------- Numba Implementation ---------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

# ---------- Utility Mappings ----------

def build_score_array(sigma_fn):
    """
    Construct a 5x5x5 score array for all possible triples of DNA alphabet.

    Parameters:
        sigma_fn (function): Scoring function taking three residues.

    Returns:
        np.ndarray: Precomputed score lookup table.
    """
    alphabet = ['A', 'C', 'G', 'T', '-']
    arr = np.zeros((5,5,5), dtype=np.int32)
    for i, a in enumerate(alphabet):
        for j, b in enumerate(alphabet):
            for k, c in enumerate(alphabet):
                arr[i,j,k] = sigma_fn(a, b, c)
    return arr


def letter_to_int_seq(seq):
    """
    Convert a DNA sequence string to an integer array representation.

    Parameters:
        seq (str): DNA string using 'A', 'C', 'G', 'T', or '-'.

    Returns:
        np.ndarray: Array of integer-encoded residues.
    """
    d = {'A':0, 'C':1, 'G':2, 'T':3, '-':4}
    return np.array([d[ch] for ch in seq], dtype=np.int32)


def int_seq_to_string(arr):
    """
    Convert an array of integer-encoded residues back to a DNA string.

    Parameters:
        arr (np.ndarray): Array of integer-encoded residues.

    Returns:
        str: Corresponding DNA string.
    """
    alpha = np.array(['A', 'C', 'G', 'T', '-'])
    return ''.join(alpha[arr])

# ---------- Numba-Accelerated DP ----------

@njit
def numba_base3d_align(A, B, C, scoremat):
    """
    Full 3D dynamic programming alignment for three sequences, including traceback.

    Parameters:
        A, B, C (np.ndarray): Integer-encoded input sequences.
        scoremat (np.ndarray): 5x5x5 precomputed scoring array.

    Returns:
        Tuple: (alignment score, aligned A, aligned B, aligned C)
    """
    m, n, k = len(A), len(B), len(C)
    D = np.full((m+1, n+1, k+1), -np.inf, dtype=np.float32)
    back = np.zeros((m+1, n+1, k+1, 3), dtype=np.int32)
    D[0, 0, 0] = 0.0

    # Initialization for faces and edges
    for i in range(1, m+1):
        D[i, 0, 0] = D[i-1, 0, 0] + scoremat[A[i-1], 4, 4]
        back[i, 0, 0] = (i-1, 0, 0)
    for j in range(1, n+1):
        D[0, j, 0] = D[0, j-1, 0] + scoremat[4, B[j-1], 4]
        back[0, j, 0] = (0, j-1, 0)
    for l in range(1, k+1):
        D[0, 0, l] = D[0, 0, l-1] + scoremat[4, 4, C[l-1]]
        back[0, 0, l] = (0, 0, l-1)

    # Faces with one dimension zero
    for i in range(1, m+1):
        for j in range(1, n+1):
            vals = [
                (D[i-1, j-1, 0] + scoremat[A[i-1], B[j-1], 4], (i-1, j-1, 0)),
                (D[i-1, j, 0]   + scoremat[A[i-1], 4, 4],      (i-1, j, 0)),
                (D[i, j-1, 0]   + scoremat[4, B[j-1], 4],      (i, j-1, 0)),
            ]
            maxv, maxb = vals[0]
            for v, b in vals:
                if v > maxv:
                    maxv, maxb = v, b
            D[i, j, 0] = maxv
            back[i, j, 0] = maxb

    for i in range(1, m+1):
        for l in range(1, k+1):
            vals = [
                (D[i-1, 0, l-1] + scoremat[A[i-1], 4, C[l-1]], (i-1, 0, l-1)),
                (D[i-1, 0, l]   + scoremat[A[i-1], 4, 4],      (i-1, 0, l)),
                (D[i, 0, l-1]   + scoremat[4, 4, C[l-1]],      (i, 0, l-1)),
            ]
            maxv, maxb = vals[0]
            for v, b in vals:
                if v > maxv:
                    maxv, maxb = v, b
            D[i, 0, l] = maxv
            back[i, 0, l] = maxb

    for j in range(1, n+1):
        for l in range(1, k+1):
            vals = [
                (D[0, j-1, l-1] + scoremat[4, B[j-1], C[l-1]], (0, j-1, l-1)),
                (D[0, j-1, l]   + scoremat[4, B[j-1], 4],      (0, j-1, l)),
                (D[0, j, l-1]   + scoremat[4, 4, C[l-1]],      (0, j, l-1)),
            ]
            maxv, maxb = vals[0]
            for v, b in vals:
                if v > maxv:
                    maxv, maxb = v, b
            D[0, j, l] = maxv
            back[0, j, l] = maxb

    # Main DP filling for all dimensions
    for i in range(1, m+1):
        for j in range(1, n+1):
            for l in range(1, k+1):
                vals = [
                    (D[i-1, j-1, l-1] + scoremat[A[i-1], B[j-1], C[l-1]], (i-1, j-1, l-1)),
                    (D[i-1, j-1, l]   + scoremat[A[i-1], B[j-1], 4],      (i-1, j-1, l)),
                    (D[i-1, j, l-1]   + scoremat[A[i-1], 4, C[l-1]],      (i-1, j, l-1)),
                    (D[i, j-1, l-1]   + scoremat[4, B[j-1], C[l-1]],      (i, j-1, l-1)),
                    (D[i-1, j, l]     + scoremat[A[i-1], 4, 4],           (i-1, j, l)),
                    (D[i, j-1, l]     + scoremat[4, B[j-1], 4],           (i, j-1, l)),
                    (D[i, j, l-1]     + scoremat[4, 4, C[l-1]],           (i, j, l-1)),
                ]
                maxv, maxb = vals[0]
                for v, b in vals:
                    if v > maxv:
                        maxv, maxb = v, b
                D[i, j, l] = maxv
                back[i, j, l] = maxb

    # Traceback to recover alignments
    i, j, l = m, n, k
    A_aln, B_aln, C_aln = [], [], []
    while (i, j, l) != (0, 0, 0):
        pi, pj, pl = back[i, j, l]
        A_aln.append(A[i-1] if i > pi else 4)
        B_aln.append(B[j-1] if j > pj else 4)
        C_aln.append(C[l-1] if l > pl else 4)
        i, j, l = pi, pj, pl
    return D[m, n, k], np.array(A_aln[::-1]), np.array(B_aln[::-1]), np.array(C_aln[::-1])


@njit
def numba_forward3dslice(A, B, C, scoremat):
    """
    Compute the forward DP slice for divide-and-conquer.

    Returns:
        np.ndarray: Last DP layer, shape (len(B)+1, len(C)+1)
    """
    m, n, k = len(A), len(B), len(C)
    PREV = np.zeros((n+1, k+1), dtype=np.float32)
    for j in range(1, n+1):
        PREV[j, 0] = PREV[j-1, 0] + scoremat[4, B[j-1], 4]
    for l in range(1, k+1):
        PREV[0, l] = PREV[0, l-1] + scoremat[4, 4, C[l-1]]
    for j in range(1, n+1):
        for l in range(1, k+1):
            PREV[j, l] = max(
                PREV[j-1, l] + scoremat[4, B[j-1], 4],
                PREV[j, l-1] + scoremat[4, 4, C[l-1]],
                PREV[j-1, l-1] + scoremat[4, B[j-1], C[l-1]]
            )
    for i in range(1, m+1):
        CURR = np.zeros((n+1, k+1), dtype=np.float32)
        CURR[0, 0] = PREV[0, 0] + scoremat[A[i-1], 4, 4]
        for l in range(1, k+1):
            CURR[0, l] = max(
                PREV[0, l-1] + scoremat[A[i-1], 4, C[l-1]],
                CURR[0, l-1] + scoremat[4, 4, C[l-1]]
            )
        for j in range(1, n+1):
            CURR[j, 0] = max(
                PREV[j-1, 0] + scoremat[A[i-1], B[j-1], 4],
                PREV[j, 0]   + scoremat[A[i-1], 4, 4],
                CURR[j-1, 0] + scoremat[4, B[j-1], 4]
            )
            for l in range(1, k+1):
                CURR[j, l] = max(
                    PREV[j-1, l-1] + scoremat[A[i-1], B[j-1], C[l-1]],
                    PREV[j-1, l]   + scoremat[A[i-1], B[j-1], 4],
                    PREV[j, l-1]   + scoremat[A[i-1], 4, C[l-1]],
                    CURR[j-1, l-1] + scoremat[4, B[j-1], C[l-1]],
                    PREV[j, l]     + scoremat[A[i-1], 4, 4],
                    CURR[j-1, l]   + scoremat[4, B[j-1], 4],
                    CURR[j, l-1]   + scoremat[4, 4, C[l-1]]
                )
        PREV = CURR
    return PREV


@njit
def numba_backward3dslice(A, B, C, scoremat):
    """
    Compute the backward DP slice for divide-and-conquer by reversing the input.

    Returns:
        np.ndarray: First DP layer from reversed computation, shape (len(B)+1, len(C)+1)
    """
    revA = A[::-1]
    revB = B[::-1]
    revC = C[::-1]
    arr = numba_forward3dslice(revA, revB, revC, scoremat)
    return arr[::-1, ::-1]



# ------------------------------------------------------------------------------
# ------------------------------ Aligner Class ---------------------------------
# ------------------------------------------------------------------------------

class Aligner:
    """
    Main class for performing three-way multiple sequence alignment using DP and
    divide-and-conquer with Numba acceleration.
    """
    def __init__(self, scoring_function: Callable[[str, str, str], int]):
        """
        Initialize the Aligner with a scoring function.

        Parameters:
            scoring_function: Function accepting three characters and returning an int score.
        """
        self.alphabet = ['A', 'C', 'G', 'T', '-']
        self.sigma = scoring_function
        self.scoremat = build_score_array(scoring_function)

    def read_fasta(self, file_path: str) -> Tuple[str, str, str]:
        """
        Read three sequences from a FASTA file.

        Parameters:
            file_path (str): Path to FASTA file.

        Returns:
            Tuple[str, str, str]: Uppercase strings of the three sequences.
        """
        records = list(SeqIO.parse(file_path, "fasta"))
        if len(records) != 3:
            raise ValueError("FASTA file must contain exactly three sequences.")
        return (str(records[0].seq).upper(),
                str(records[1].seq).upper(),
                str(records[2].seq).upper())

    def BASE3DALIGN(self, A: str, B: str, C: str) -> Tuple[int, str, str, str]:
        """
        Compute full 3D DP alignment (base case).

        Returns:
            Tuple: (score, aligned A, aligned B, aligned C)
        """
        Aint = letter_to_int_seq(A)
        Bint = letter_to_int_seq(B)
        Cint = letter_to_int_seq(C)
        score, A_aln, B_aln, C_aln = numba_base3d_align(Aint, Bint, Cint, self.scoremat)
        return int(score), int_seq_to_string(A_aln), int_seq_to_string(B_aln), int_seq_to_string(C_aln)

    def FORWARD3DSLICE(self, A: str, B: str, C: str):
        """
        Compute the forward DP slice for divide-and-conquer.

        Returns:
            np.ndarray: DP slice (len(B)+1, len(C)+1)
        """
        Aint = letter_to_int_seq(A)
        Bint = letter_to_int_seq(B)
        Cint = letter_to_int_seq(C)
        return numba_forward3dslice(Aint, Bint, Cint, self.scoremat)

    def BACKWARD3DSLICE(self, A: str, B: str, C: str):
        """
        Compute the backward DP slice for divide-and-conquer.

        Returns:
            np.ndarray: DP slice (len(B)+1, len(C)+1)
        """
        Aint = letter_to_int_seq(A)
        Bint = letter_to_int_seq(B)
        Cint = letter_to_int_seq(C)
        return numba_backward3dslice(Aint, Bint, Cint, self.scoremat)

    def DC_THREEWAY_ALIGN(self, A: str, B: str, C: str, threshold: int = 15):
        """
        Recursively perform divide-and-conquer alignment. Fall back to base DP for small subproblems.

        Returns:
            Tuple: (score, aligned A, aligned B, aligned C)
        """
        if min(len(A), len(B), len(C)) <= threshold:
            return self.BASE3DALIGN(A, B, C)
        i = len(A) // 2
        F = self.FORWARD3DSLICE(A[:i], B, C)
        R = self.BACKWARD3DSLICE(A[i:], B, C)
        j_star, l_star = np.unravel_index(np.argmax(F + R), F.shape)
        left_score, A1, B1, C1 = self.DC_THREEWAY_ALIGN(A[:i], B[:j_star], C[:l_star], threshold)
        right_score, A2, B2, C2 = self.DC_THREEWAY_ALIGN(A[i:], B[j_star:], C[l_star:], threshold)
        return (left_score + right_score, A1 + A2, B1 + B2, C1 + C2)

    def run_from_fasta(self, fasta_path: str, use_divide_and_conquer: bool = True):
        """
        Run the full alignment pipeline from a FASTA file and print results.

        Parameters:
            fasta_path (str): Path to FASTA file with three sequences.
            use_divide_and_conquer (bool): Use DC algorithm if True, else use full DP.
        """
        print(f'\n\n-------------- Sequence {fasta_path} ---------------\n')
        stop_event = threading.Event()
        peak_rss = [psutil.Process(os.getpid()).memory_info().rss]
        monitor_thread = threading.Thread(target=monitor_peak_memory, args=(stop_event, peak_rss))
        monitor_thread.start()

        A, B, C = self.read_fasta(fasta_path)
        print(f"Lengths: A={len(A)}, B={len(B)}, C={len(C)}")
        t0 = time.time()
        if use_divide_and_conquer:
            score, A2, B2, C2 = self.DC_THREEWAY_ALIGN(A, B, C)
        else:
            score, A2, B2, C2 = self.BASE3DALIGN(A, B, C)
        t1 = time.time()

        stop_event.set()
        monitor_thread.join()
        mem_used_mb = peak_rss[0] / (1024 * 1024)

        # Count columns with perfect non-gap matches
        match_count = 0
        total_len = len(A2)
        for x, y, z in zip(A2, B2, C2):
            if x == y == z and x != '-':
                match_count += 1

        print("\nAlignment Statistics\n" + "-"*40)
        print(f"{'Metric':<35} {'Value'}")
        print("-"*40)
        print(f"{'Score':<35} {score}")
        print(f"{'Time (s)':<35} {t1-t0:.2f}")
        print(f"{'Peak memory used (MB)':<35} {mem_used_mb:.2f}")
        print(f"{'Exact non-gap matches':<35} {match_count}")
        print(f"{'Alignment length':<35} {total_len}")
        print("-"*40)

        print("\nAlignment:")
        print("A:", A2)
        print("B:", B2)
        print("C:", C2)



# --------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------- MAIN FUNCTION --------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------

def main():

    # --------------------- INPUT DNA SEQUENCES -------------------------- #
    # Example DNA string input (repeat/edit as needed for each sequence)                # Input your strings here
    raw_input1 = """ACGTAGGTACGTACGTTCGTACGTACGTACGTACGTAC"""
    raw_input2 = """ACGTCGGTACGTACGTTGGTACGTACGTCGGTACGTAA"""
    raw_input3 = """ACGTAGGTTCGTACGTACGTACGTCCGTACGTACGTGC"""

    # # Generate three test .txt files for alignment                                    # Generate .txt files for the dna sequences
    dnagen(raw_input1, output_file='test1.txt')
    dnagen(raw_input2, output_file='test2.txt')
    dnagen(raw_input3, output_file='test3.txt')

    # # Combine into FASTA file for Aligner                                             # Generate fasta file from the sequences.
    fastagen(['test1.txt', 'test2.txt', 'test3.txt'], output_fasta='example.fasta')

    # ------------------------------------------- #
    # --------------- ALIGNMENT ----------------- #
    # ------------------------------------------- #

    aligner = Aligner(scoring_function=blast_sigma)                                     # Create an aligner instance 
    aligner.run_from_fasta('example.fasta', use_divide_and_conquer=True)                # Run the alignment



if __name__ == "__main__":
    main()