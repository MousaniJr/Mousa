"""
Continuous Learning System

Implements automated feedback collection, validation, and retraining pipeline
Allows the model to improve over time based on real-world usage
"""

import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UserFeedback:
    """User feedback on a model response"""
    feedback_id: str
    prompt: str
    response: str
    rating: int  # 1-5 stars
    was_helpful: bool
    was_correct: bool
    was_safe: bool
    user_correction: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ValidationResult:
    """Result of automated validation"""
    is_valid: bool
    quality_score: float  # 0.0-1.0
    issues: List[str]
    passed_safety: bool
    passed_correctness: bool
    passed_style: bool


class FeedbackCollector:
    """
    Collects and stores user feedback anonymously
    """

    def __init__(self, storage_dir: str = "data/feedback"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_buffer: List[UserFeedback] = []

    def add_feedback(
        self,
        prompt: str,
        response: str,
        rating: int,
        was_helpful: bool,
        was_correct: bool,
        was_safe: bool,
        user_correction: Optional[str] = None
    ) -> str:
        """
        Add user feedback

        Args:
            prompt: User's original prompt
            response: Model's response
            rating: User rating (1-5)
            was_helpful: Whether response was helpful
            was_correct: Whether response was correct
            was_safe: Whether response was safe
            user_correction: Optional user-provided correction

        Returns:
            Feedback ID
        """
        # Generate feedback ID
        feedback_id = hashlib.sha256(
            f"{prompt}{response}{datetime.now()}".encode()
        ).hexdigest()[:16]

        # Create feedback object
        feedback = UserFeedback(
            feedback_id=feedback_id,
            prompt=prompt,
            response=response,
            rating=rating,
            was_helpful=was_helpful,
            was_correct=was_correct,
            was_safe=was_safe,
            user_correction=user_correction
        )

        # Add to buffer
        self.feedback_buffer.append(feedback)

        # Persist if buffer is large
        if len(self.feedback_buffer) >= 100:
            self.flush()

        logger.info(f"Feedback {feedback_id} collected: rating={rating}")
        return feedback_id

    def flush(self):
        """Persist feedback buffer to disk"""
        if not self.feedback_buffer:
            return

        # Save to daily file
        date_str = datetime.now().strftime("%Y-%m-%d")
        feedback_file = self.storage_dir / f"feedback_{date_str}.jsonl"

        with open(feedback_file, 'a') as f:
            for feedback in self.feedback_buffer:
                f.write(json.dumps(asdict(feedback), default=str) + '\n')

        logger.info(f"Flushed {len(self.feedback_buffer)} feedback items")
        self.feedback_buffer.clear()

    def load_feedback(self, days: int = 7) -> List[UserFeedback]:
        """
        Load feedback from recent days

        Args:
            days: Number of days to load

        Returns:
            List of feedback items
        """
        feedback_items = []
        cutoff_date = datetime.now() - timedelta(days=days)

        for feedback_file in self.storage_dir.glob("feedback_*.jsonl"):
            with open(feedback_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])

                    if data['timestamp'] >= cutoff_date:
                        feedback_items.append(UserFeedback(**data))

        logger.info(f"Loaded {len(feedback_items)} feedback items from last {days} days")
        return feedback_items

    def get_statistics(self, feedback_items: List[UserFeedback]) -> Dict:
        """Get feedback statistics"""
        if not feedback_items:
            return {}

        stats = {
            "total_feedback": len(feedback_items),
            "avg_rating": sum(f.rating for f in feedback_items) / len(feedback_items),
            "helpful_rate": sum(f.was_helpful for f in feedback_items) / len(feedback_items),
            "correct_rate": sum(f.was_correct for f in feedback_items) / len(feedback_items),
            "safe_rate": sum(f.was_safe for f in feedback_items) / len(feedback_items),
            "has_correction": sum(1 for f in feedback_items if f.user_correction) / len(feedback_items)
        }

        return stats


class AutomatedValidator:
    """
    Validates model responses before including in training data
    """

    def __init__(self):
        from src.data_engine.validators.security_validator import SecurityValidator
        from src.data_engine.processors.preprocess import QualityFilter

        self.security_validator = SecurityValidator(strict_mode=False)
        self.quality_filter = QualityFilter()

    def validate_response(
        self,
        prompt: str,
        response: str,
        language: str = "Python"
    ) -> ValidationResult:
        """
        Validate a model response

        Args:
            prompt: Original prompt
            response: Model response
            language: Programming language

        Returns:
            ValidationResult
        """
        issues = []
        passed_safety = True
        passed_correctness = True
        passed_style = True

        # 1. Security validation
        is_safe, security_issues = self.security_validator.scan_file(response, language)
        if not is_safe:
            passed_safety = False
            issues.extend([f"Security: {issue.description}" for issue in security_issues])

        # 2. Code quality check (if response contains code)
        code_blocks = self._extract_code_blocks(response)
        if code_blocks:
            for code in code_blocks:
                quality_score = self.quality_filter.compute_quality_score(code, language)
                if quality_score < 0.5:
                    passed_style = False
                    issues.append(f"Code quality score too low: {quality_score:.2f}")

        # 3. Correctness heuristics
        # Check for common error patterns
        error_patterns = [
            "error:", "exception:", "failed:", "incorrect:",
            "TODO", "FIXME", "not implemented"
        ]
        for pattern in error_patterns:
            if pattern in response.lower():
                issues.append(f"Response contains '{pattern}'")

        # 4. Compute overall quality score
        quality_score = 1.0
        if not passed_safety:
            quality_score -= 0.5
        if not passed_style:
            quality_score -= 0.2
        if not passed_correctness:
            quality_score -= 0.3

        quality_score = max(0.0, quality_score)

        # 5. Determine if valid
        is_valid = passed_safety and quality_score >= 0.5

        return ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            issues=issues,
            passed_safety=passed_safety,
            passed_correctness=passed_correctness,
            passed_style=passed_style
        )

    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from markdown text"""
        import re
        # Match ```language\ncode\n``` blocks
        pattern = r'```(?:\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches


class ContinuousLearningPipeline:
    """
    Main continuous learning pipeline
    Coordinates feedback collection, validation, and retraining
    """

    def __init__(
        self,
        feedback_dir: str = "data/feedback",
        training_output_dir: str = "data/continuous_learning",
        min_feedback_for_retrain: int = 1000
    ):
        self.feedback_collector = FeedbackCollector(feedback_dir)
        self.validator = AutomatedValidator()
        self.training_output_dir = Path(training_output_dir)
        self.training_output_dir.mkdir(parents=True, exist_ok=True)
        self.min_feedback_for_retrain = min_feedback_for_retrain

    def collect_and_validate_feedback(self, days: int = 7) -> List[Dict]:
        """
        Collect recent feedback and validate for training

        Args:
            days: Number of days of feedback to process

        Returns:
            List of validated training examples
        """
        logger.info(f"Processing feedback from last {days} days...")

        # Load feedback
        feedback_items = self.feedback_collector.load_feedback(days)

        if len(feedback_items) < self.min_feedback_for_retrain:
            logger.warning(
                f"Only {len(feedback_items)} feedback items, "
                f"need {self.min_feedback_for_retrain} for retraining"
            )

        # Get statistics
        stats = self.feedback_collector.get_statistics(feedback_items)
        logger.info(f"Feedback stats: {json.dumps(stats, indent=2)}")

        # Filter and validate
        validated_examples = []

        for feedback in feedback_items:
            # Only include high-quality, positive feedback
            if feedback.rating >= 4 and feedback.was_helpful and feedback.was_correct:
                # Validate response
                validation = self.validator.validate_response(
                    feedback.prompt,
                    feedback.response
                )

                if validation.is_valid:
                    validated_examples.append({
                        "prompt": feedback.prompt,
                        "response": feedback.response,
                        "quality_score": validation.quality_score,
                        "user_rating": feedback.rating,
                        "timestamp": feedback.timestamp.isoformat()
                    })

            # If user provided correction, use that
            elif feedback.user_correction:
                validated_examples.append({
                    "prompt": feedback.prompt,
                    "response": feedback.user_correction,
                    "quality_score": 1.0,
                    "user_rating": 5,  # User corrections are high quality
                    "timestamp": feedback.timestamp.isoformat(),
                    "is_correction": True
                })

        logger.info(f"Validated {len(validated_examples)} training examples")
        return validated_examples

    def prepare_training_data(self, validated_examples: List[Dict]) -> str:
        """
        Prepare validated examples for fine-tuning

        Args:
            validated_examples: List of validated examples

        Returns:
            Path to training data file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.training_output_dir / f"training_data_{timestamp}.jsonl"

        with open(output_file, 'w') as f:
            for example in validated_examples:
                f.write(json.dumps(example) + '\n')

        logger.info(f"Saved {len(validated_examples)} examples to {output_file}")
        return str(output_file)

    def should_trigger_retraining(self) -> bool:
        """
        Determine if enough new data exists to trigger retraining

        Returns:
            True if retraining should be triggered
        """
        # Get recent feedback
        recent_feedback = self.feedback_collector.load_feedback(days=7)

        # Check if we have enough high-quality feedback
        high_quality_count = sum(
            1 for f in recent_feedback
            if f.rating >= 4 and (f.was_helpful or f.user_correction)
        )

        logger.info(f"High-quality feedback count: {high_quality_count}")

        return high_quality_count >= self.min_feedback_for_retrain

    def run_continuous_learning_cycle(self):
        """
        Run a complete continuous learning cycle
        """
        logger.info("Starting continuous learning cycle...")

        # 1. Check if retraining is needed
        if not self.should_trigger_retraining():
            logger.info("Not enough feedback for retraining yet")
            return

        # 2. Collect and validate feedback
        validated_examples = self.collect_and_validate_feedback(days=7)

        if not validated_examples:
            logger.warning("No validated examples found")
            return

        # 3. Prepare training data
        training_file = self.prepare_training_data(validated_examples)

        # 4. Trigger retraining (would integrate with training pipeline)
        logger.info(f"Retraining triggered with {len(validated_examples)} examples")
        logger.info(f"Training data: {training_file}")

        # In production, this would launch a fine-tuning job
        # For now, just log the action
        logger.info("✅ Continuous learning cycle complete")

        return training_file


def create_retraining_schedule():
    """
    Define automated retraining schedule

    Returns schedule configuration
    """
    schedule = {
        "weekly_finetuning": {
            "frequency": "weekly",
            "day": "Sunday",
            "time": "02:00",
            "min_examples": 1000,
            "description": "Weekly fine-tuning on validated user feedback"
        },
        "monthly_full_retrain": {
            "frequency": "monthly",
            "day": 1,
            "time": "02:00",
            "min_examples": 10000,
            "description": "Monthly full retraining with accumulated data"
        },
        "realtime_validation": {
            "frequency": "continuous",
            "description": "Real-time validation of user feedback"
        }
    }

    return schedule


if __name__ == "__main__":
    # Example usage
    pipeline = ContinuousLearningPipeline()

    # Simulate adding feedback
    feedback_collector = pipeline.feedback_collector

    feedback_collector.add_feedback(
        prompt="Write a Python function to reverse a string",
        response="def reverse_string(s):\n    return s[::-1]",
        rating=5,
        was_helpful=True,
        was_correct=True,
        was_safe=True
    )

    # Run learning cycle
    pipeline.run_continuous_learning_cycle()

    # Print schedule
    schedule = create_retraining_schedule()
    print("\nRetraining Schedule:")
    print(json.dumps(schedule, indent=2))
