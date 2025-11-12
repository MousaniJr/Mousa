"""
Data Preprocessing Pipeline

Handles: Deduplication, tokenization, quality filtering, augmentation
"""

import hashlib
import re
from typing import List, Dict, Set, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
import json
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CodeSample:
    """Represents a processed code sample"""
    content: str
    language: str
    file_path: str
    hash: str
    quality_score: float
    metadata: Dict


class Deduplicator:
    """
    Removes duplicate and near-duplicate code samples
    """

    def __init__(self, similarity_threshold: float = 0.9):
        """
        Initialize deduplicator

        Args:
            similarity_threshold: Similarity threshold for near-duplicates (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold
        self.seen_hashes: Set[str] = set()
        self.exact_duplicates = 0
        self.near_duplicates = 0

    @staticmethod
    def compute_hash(content: str) -> str:
        """
        Compute SHA-256 hash of content

        Args:
            content: Code content

        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    @staticmethod
    def normalize_code(code: str) -> str:
        """
        Normalize code for near-duplicate detection

        - Remove comments
        - Normalize whitespace
        - Remove variable names (replace with placeholders)

        Args:
            code: Original code

        Returns:
            Normalized code
        """
        # Remove single-line comments
        code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
        code = re.sub(r'#.*?$', '', code, flags=re.MULTILINE)

        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)

        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)

        # Remove string literals
        code = re.sub(r'"[^"]*"', '""', code)
        code = re.sub(r"'[^']*'", "''", code)

        return code.strip().lower()

    def is_duplicate(self, content: str) -> bool:
        """
        Check if content is a duplicate

        Args:
            content: Code content

        Returns:
            True if duplicate, False otherwise
        """
        # Exact duplicate check
        exact_hash = self.compute_hash(content)
        if exact_hash in self.seen_hashes:
            self.exact_duplicates += 1
            return True

        # Near-duplicate check (using normalized content)
        normalized = self.normalize_code(content)
        normalized_hash = self.compute_hash(normalized)

        if normalized_hash in self.seen_hashes:
            self.near_duplicates += 1
            return True

        # Store both hashes
        self.seen_hashes.add(exact_hash)
        self.seen_hashes.add(normalized_hash)

        return False

    def get_stats(self) -> Dict[str, int]:
        """Get deduplication statistics"""
        return {
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "total_duplicates": self.exact_duplicates + self.near_duplicates
        }


class QualityFilter:
    """
    Filters low-quality code samples
    """

    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 10000,
        min_quality_score: float = 0.5
    ):
        """
        Initialize quality filter

        Args:
            min_length: Minimum code length (characters)
            max_length: Maximum code length (characters)
            min_quality_score: Minimum quality score (0.0-1.0)
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_quality_score = min_quality_score

    def compute_quality_score(self, code: str, language: str) -> float:
        """
        Compute quality score for code sample

        Criteria:
        - Proper indentation
        - Comment density
        - Variable naming quality
        - Code structure

        Args:
            code: Code content
            language: Programming language

        Returns:
            Quality score (0.0-1.0)
        """
        score = 1.0
        lines = code.split('\n')

        # Penalty for very short or very long code
        if len(code) < self.min_length:
            score -= 0.3
        if len(code) > self.max_length:
            score -= 0.2

        # Check for proper indentation
        indent_levels = []
        for line in lines:
            if line.strip():
                leading_spaces = len(line) - len(line.lstrip())
                indent_levels.append(leading_spaces)

        if len(set(indent_levels)) > 1:  # Has indentation
            score += 0.1
        else:
            score -= 0.2  # No indentation (likely low quality)

        # Check for comments (indicates documentation)
        comment_patterns = [r'#', r'//', r'/\*', r'"""', r"'''"]
        has_comments = any(re.search(pattern, code) for pattern in comment_patterns)
        if has_comments:
            score += 0.1

        # Check for meaningful variable names (not single letters)
        single_letter_vars = len(re.findall(r'\b[a-z]\s*=', code))
        if single_letter_vars > 5:
            score -= 0.1

        # Check for TODO/FIXME (indicates incomplete code)
        if re.search(r'\b(TODO|FIXME|XXX|HACK)\b', code, re.IGNORECASE):
            score -= 0.1

        # Check for excessive blank lines
        blank_lines = sum(1 for line in lines if not line.strip())
        if blank_lines / max(len(lines), 1) > 0.3:
            score -= 0.1

        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))

    def passes_filter(self, code: str, language: str) -> bool:
        """
        Check if code passes quality filter

        Args:
            code: Code content
            language: Programming language

        Returns:
            True if passes filter, False otherwise
        """
        quality = self.compute_quality_score(code, language)
        return quality >= self.min_quality_score


class DataPreprocessor:
    """
    Main preprocessing pipeline
    """

    def __init__(
        self,
        input_dir: str = "data/raw/code",
        output_dir: str = "data/processed",
        deduplicate: bool = True,
        quality_filter: bool = True
    ):
        """
        Initialize preprocessor

        Args:
            input_dir: Directory containing raw code
            output_dir: Directory for processed data
            deduplicate: Whether to remove duplicates
            quality_filter: Whether to apply quality filtering
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.deduplicator = Deduplicator() if deduplicate else None
        self.quality_filter = QualityFilter() if quality_filter else None

        self.stats = defaultdict(int)

    def process_file(self, file_path: Path, language: str) -> Optional[CodeSample]:
        """
        Process a single file

        Args:
            file_path: Path to code file
            language: Programming language

        Returns:
            CodeSample if valid, None if filtered out
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            self.stats['total_files'] += 1

            # Skip empty files
            if not content.strip():
                self.stats['empty_files'] += 1
                return None

            # Deduplication
            if self.deduplicator and self.deduplicator.is_duplicate(content):
                self.stats['duplicates'] += 1
                return None

            # Quality filtering
            if self.quality_filter:
                quality_score = self.quality_filter.compute_quality_score(content, language)
                if not self.quality_filter.passes_filter(content, language):
                    self.stats['low_quality'] += 1
                    return None
            else:
                quality_score = 1.0

            # Create code sample
            sample = CodeSample(
                content=content,
                language=language,
                file_path=str(file_path),
                hash=hashlib.sha256(content.encode()).hexdigest(),
                quality_score=quality_score,
                metadata={
                    "size": len(content),
                    "lines": len(content.split('\n'))
                }
            )

            self.stats['accepted_files'] += 1
            return sample

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.stats['errors'] += 1
            return None

    def process_directory(self, language_dir: Path) -> List[CodeSample]:
        """
        Process all files in a language directory

        Args:
            language_dir: Directory containing code files for one language

        Returns:
            List of valid code samples
        """
        language = language_dir.name
        logger.info(f"Processing {language} files...")

        samples = []
        file_extensions = {
            "Python": ".py",
            "JavaScript": ".js",
            "TypeScript": ".ts",
            "Java": ".java",
            "C++": [".cpp", ".cc", ".cxx"],
            "C": ".c",
            "Go": ".go",
            "Rust": ".rs",
            "Swift": ".swift"
        }

        # Get file extension(s) for this language
        extensions = file_extensions.get(language, [])
        if isinstance(extensions, str):
            extensions = [extensions]

        # Process all matching files
        for ext in extensions:
            for file_path in language_dir.rglob(f"*{ext}"):
                sample = self.process_file(file_path, language)
                if sample:
                    samples.append(sample)

        logger.info(f"Processed {len(samples)} valid {language} samples")
        return samples

    def save_samples(self, samples: List[CodeSample], language: str):
        """
        Save processed samples to disk

        Args:
            samples: List of code samples
            language: Programming language
        """
        output_file = self.output_dir / f"{language.lower()}_processed.jsonl"

        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                record = {
                    "content": sample.content,
                    "language": sample.language,
                    "quality_score": sample.quality_score,
                    "hash": sample.hash,
                    "metadata": sample.metadata
                }
                f.write(json.dumps(record) + '\n')

        logger.info(f"Saved {len(samples)} samples to {output_file}")

    def process_all(self):
        """
        Process all languages in the input directory
        """
        for language_dir in self.input_dir.iterdir():
            if language_dir.is_dir():
                samples = self.process_directory(language_dir)
                if samples:
                    self.save_samples(samples, language_dir.name)

        # Print statistics
        logger.info("\n=== Preprocessing Statistics ===")
        for key, value in self.stats.items():
            logger.info(f"{key}: {value}")

        if self.deduplicator:
            dup_stats = self.deduplicator.get_stats()
            for key, value in dup_stats.items():
                logger.info(f"{key}: {value}")


def main():
    """Example usage"""
    preprocessor = DataPreprocessor(
        input_dir="data/raw/code",
        output_dir="data/processed",
        deduplicate=True,
        quality_filter=True
    )

    preprocessor.process_all()


if __name__ == "__main__":
    main()
