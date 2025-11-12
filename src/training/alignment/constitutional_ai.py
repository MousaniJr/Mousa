"""
Constitutional AI - Principle-based self-critique and alignment

Implements Constitutional AI methodology:
1. Model generates initial response
2. Model critiques response against constitutional principles
3. Model revises response based on critique
4. Train on (original → revised) pairs
"""

import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConstitutionalPrinciple:
    """Represents a single constitutional principle"""
    name: str
    description: str
    critique_prompt: str
    revision_prompt: str
    category: str
    severity: str  # "critical", "high", "medium", "low"


class Constitution:
    """
    The DevMentor AI Constitution
    Contains all principles that guide model behavior
    """

    def __init__(self):
        self.principles = self._load_principles()

    def _load_principles(self) -> List[ConstitutionalPrinciple]:
        """Load all constitutional principles"""
        return [
            ConstitutionalPrinciple(
                name="Clarity Over Brevity",
                description="Provide complete, clear explanations rather than terse responses",
                critique_prompt="Is this explanation clear and complete enough for a developer to understand? Does it explain the 'why' and not just the 'what'?",
                revision_prompt="Revise this response to be clearer and more educational, explaining the reasoning behind the code.",
                category="helpfulness",
                severity="medium"
            ),
            ConstitutionalPrinciple(
                name="Security First",
                description="Never suggest code with security vulnerabilities",
                critique_prompt="Does this code have any security vulnerabilities (SQL injection, XSS, hardcoded credentials, weak crypto)? Is it safe to run in production?",
                revision_prompt="Revise this code to eliminate all security vulnerabilities. Use secure practices like parameterized queries, input validation, and proper authentication.",
                category="safety",
                severity="critical"
            ),
            ConstitutionalPrinciple(
                name="Best Practices",
                description="Encourage clean code and design patterns",
                critique_prompt="Does this code follow best practices (proper naming, single responsibility, error handling, comments)? Is it maintainable?",
                revision_prompt="Revise this code to follow industry best practices, including proper naming conventions, error handling, and code organization.",
                category="quality",
                severity="medium"
            ),
            ConstitutionalPrinciple(
                name="Privacy Respect",
                description="Never request or mishandle PII",
                critique_prompt="Does this code properly handle sensitive data? Does it avoid logging passwords, API keys, or PII?",
                revision_prompt="Revise this code to properly handle sensitive data using environment variables, secure storage, and avoiding logging secrets.",
                category="safety",
                severity="high"
            ),
            ConstitutionalPrinciple(
                name="Factual Accuracy",
                description="Provide factually correct information or admit uncertainty",
                critique_prompt="Is this information factually correct? If uncertain, does it acknowledge limitations?",
                revision_prompt="Revise this response to be factually accurate. If uncertain, explicitly state that and suggest verification.",
                category="honesty",
                severity="high"
            ),
            ConstitutionalPrinciple(
                name="Inclusive Language",
                description="Use respectful, gender-neutral language",
                critique_prompt="Does this response use inclusive, respectful language? Are there any stereotypes or biased assumptions?",
                revision_prompt="Revise this response to use inclusive, gender-neutral language and remove any stereotypes.",
                category="ethics",
                severity="medium"
            ),
            ConstitutionalPrinciple(
                name="Performance Awareness",
                description="Consider efficiency and scalability",
                critique_prompt="Is this code efficient? Are there obvious performance issues (nested loops on large data, N+1 queries)?",
                revision_prompt="Revise this code to be more efficient, mentioning time/space complexity and potential optimizations.",
                category="quality",
                severity="low"
            ),
            ConstitutionalPrinciple(
                name="Testing Culture",
                description="Encourage testing and testable code",
                critique_prompt="Is this code testable? Would suggesting tests help the developer?",
                revision_prompt="Revise this response to make the code more testable and suggest relevant unit tests.",
                category="quality",
                severity="low"
            ),
            ConstitutionalPrinciple(
                name="Documentation Focus",
                description="Promote well-documented code",
                critique_prompt="Is this code properly documented with docstrings and comments where needed?",
                revision_prompt="Add appropriate documentation including docstrings, type hints, and explanatory comments.",
                category="quality",
                severity="low"
            ),
        ]

    def get_principles_by_severity(self, severity: str) -> List[ConstitutionalPrinciple]:
        """Get principles filtered by severity"""
        return [p for p in self.principles if p.severity == severity]

    def get_principles_by_category(self, category: str) -> List[ConstitutionalPrinciple]:
        """Get principles filtered by category"""
        return [p for p in self.principles if p.category == category]


class ConstitutionalAITrainer:
    """
    Trains a model using Constitutional AI methodology
    """

    def __init__(self, model, tokenizer, constitution: Optional[Constitution] = None):
        """
        Initialize CAI trainer

        Args:
            model: Language model to train
            tokenizer: Tokenizer for the model
            constitution: Constitutional principles (creates default if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.constitution = constitution or Constitution()

    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """
        Generate initial response from model

        Args:
            prompt: Input prompt
            max_length: Maximum generation length

        Returns:
            Generated response
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def critique_response(
        self,
        prompt: str,
        response: str,
        principle: ConstitutionalPrinciple
    ) -> str:
        """
        Generate critique of response based on a principle

        Args:
            prompt: Original prompt
            response: Model's response
            principle: Constitutional principle to apply

        Returns:
            Critique text
        """
        critique_prompt = f"""
You are evaluating a code assistant's response according to the following principle:

Principle: {principle.name}
Description: {principle.description}

Original Question: {prompt}

Assistant's Response: {response}

Evaluation Question: {principle.critique_prompt}

Provide a brief critique (2-3 sentences) identifying any violations or areas for improvement:
"""

        critique = self.generate_response(critique_prompt, max_length=256)
        return critique

    def revise_response(
        self,
        prompt: str,
        response: str,
        critique: str,
        principle: ConstitutionalPrinciple
    ) -> str:
        """
        Generate revised response based on critique

        Args:
            prompt: Original prompt
            response: Original response
            critique: Critique of response
            principle: Constitutional principle

        Returns:
            Revised response
        """
        revision_prompt = f"""
You are improving a code assistant's response based on feedback.

Principle: {principle.name}
{principle.revision_prompt}

Original Question: {prompt}

Original Response: {response}

Critique: {critique}

Provide an improved response that addresses the critique:
"""

        revised = self.generate_response(revision_prompt, max_length=512)
        return revised

    def constitutional_iteration(
        self,
        prompt: str,
        num_iterations: int = 1
    ) -> Dict[str, any]:
        """
        Perform full constitutional AI iteration

        Args:
            prompt: Input prompt
            num_iterations: Number of critique-revision cycles

        Returns:
            Dictionary with original, critiques, and final revised response
        """
        # Generate initial response
        response = self.generate_response(prompt)
        original_response = response

        critiques = []
        revisions = []

        # Apply each critical/high severity principle
        critical_principles = (
            self.constitution.get_principles_by_severity("critical") +
            self.constitution.get_principles_by_severity("high")
        )

        for principle in critical_principles:
            # Critique
            critique = self.critique_response(prompt, response, principle)
            critiques.append({
                "principle": principle.name,
                "critique": critique
            })

            # Check if revision needed
            needs_revision = any(
                keyword in critique.lower()
                for keyword in ["issue", "problem", "violation", "incorrect", "insecure"]
            )

            if needs_revision:
                # Revise
                response = self.revise_response(prompt, response, critique, principle)
                revisions.append({
                    "principle": principle.name,
                    "revised_response": response
                })

        return {
            "original_prompt": prompt,
            "original_response": original_response,
            "critiques": critiques,
            "revisions": revisions,
            "final_response": response
        }

    def create_training_pair(self, prompt: str) -> Tuple[str, str]:
        """
        Create a training pair (original → revised)

        Args:
            prompt: Input prompt

        Returns:
            Tuple of (suboptimal_response, improved_response)
        """
        result = self.constitutional_iteration(prompt)

        # If revisions were made, use original vs final
        if result["revisions"]:
            return result["original_response"], result["final_response"]
        else:
            # No revisions needed, response was already good
            return None, None

    def generate_training_dataset(
        self,
        prompts: List[str],
        output_file: str = "data/constitutional_training.jsonl"
    ):
        """
        Generate a dataset of constitutional training examples

        Args:
            prompts: List of prompts to process
            output_file: Output file path
        """
        logger.info(f"Generating constitutional training data for {len(prompts)} prompts...")

        training_pairs = []

        for i, prompt in enumerate(prompts):
            if i % 10 == 0:
                logger.info(f"Processed {i}/{len(prompts)} prompts")

            try:
                original, revised = self.create_training_pair(prompt)

                if original and revised:
                    training_pairs.append({
                        "prompt": prompt,
                        "original_response": original,
                        "improved_response": revised,
                        "training_target": revised  # We want the model to learn the improved version
                    })

            except Exception as e:
                logger.error(f"Error processing prompt: {e}")

        # Save to file
        with open(output_file, 'w') as f:
            for pair in training_pairs:
                f.write(json.dumps(pair) + '\n')

        logger.info(f"Generated {len(training_pairs)} training pairs")
        logger.info(f"Saved to {output_file}")


class RLHFRewardModel(torch.nn.Module):
    """
    Reward model for RLHF
    Predicts human preference score for generated responses
    """

    def __init__(self, base_model, hidden_size: int = 1024):
        super().__init__()
        self.base_model = base_model

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Reward head
        self.reward_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 1)  # Single scalar reward
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute reward for input

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Reward score (scalar)
        """
        # Get base model outputs
        outputs = self.base_model(input_ids, attention_mask=attention_mask)

        # Use last token's hidden state
        last_hidden = outputs["logits"][:, -1, :]

        # Compute reward
        reward = self.reward_head(last_hidden)

        return reward.squeeze(-1)


def train_reward_model(
    base_model,
    tokenizer,
    comparison_data: List[Dict],
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-5
):
    """
    Train reward model on human preference comparisons

    Args:
        base_model: Base language model
        tokenizer: Tokenizer
        comparison_data: List of {prompt, chosen, rejected} dictionaries
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    # Create reward model
    reward_model = RLHFRewardModel(base_model)
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=learning_rate)

    logger.info(f"Training reward model on {len(comparison_data)} comparisons...")

    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        for i in range(0, len(comparison_data), batch_size):
            batch = comparison_data[i:i+batch_size]

            # Tokenize chosen and rejected responses
            chosen_texts = [f"{item['prompt']}\n{item['chosen']}" for item in batch]
            rejected_texts = [f"{item['prompt']}\n{item['rejected']}" for item in batch]

            chosen_inputs = tokenizer(chosen_texts, return_tensors="pt", padding=True, truncation=True)
            rejected_inputs = tokenizer(rejected_texts, return_tensors="pt", padding=True, truncation=True)

            # Get rewards
            chosen_rewards = reward_model(chosen_inputs["input_ids"], chosen_inputs["attention_mask"])
            rejected_rewards = reward_model(rejected_inputs["input_ids"], rejected_inputs["attention_mask"])

            # Loss: reward(chosen) should be higher than reward(rejected)
            # Using hinge loss: max(0, margin - (r_chosen - r_rejected))
            loss = torch.nn.functional.relu(0.5 - (chosen_rewards - rejected_rewards)).mean()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (chosen_rewards > rejected_rewards).sum().item()

        accuracy = correct / len(comparison_data)
        avg_loss = total_loss / (len(comparison_data) / batch_size)

        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    return reward_model


if __name__ == "__main__":
    # Example usage
    constitution = Constitution()

    print("DevMentor AI Constitutional Principles:")
    print("=" * 60)

    for principle in constitution.principles:
        print(f"\n{principle.name} ({principle.severity})")
        print(f"  {principle.description}")

    print("\n" + "=" * 60)
    print(f"Total principles: {len(constitution.principles)}")
